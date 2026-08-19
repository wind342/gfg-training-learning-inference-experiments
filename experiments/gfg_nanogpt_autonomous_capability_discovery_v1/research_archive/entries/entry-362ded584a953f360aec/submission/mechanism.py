"""Sealed prefix-conditioned formation and stability mechanism.

The executable state is intentionally small: formation charge and velocity,
plus the phase, recovery memory, and accumulated clipped-impulse burden of a
relaxation oscillator.  It never reads a graph suffix during ``forecast``.
"""

import math


def _rows(prefix):
    rows = list(prefix.evaluations())
    rows.sort(key=lambda row: int(row["optimizer_step"]))
    return rows


def _median(values, default):
    if not values:
        return float(default)
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _observed_transition(evaluations):
    for index in range(max(0, len(evaluations) - 2)):
        window = evaluations[index:index + 3]
        if len(window) == 3 and all(
            float(row["train_accuracy"]) >= 0.99
            and float(row["validation_accuracy"]) >= 0.90
            for row in window
        ):
            earlier_low = any(
                float(row["validation_accuracy"]) <= 0.30
                for row in evaluations[:index]
            )
            if earlier_low:
                return int(window[0]["optimizer_step"])
    return None


def _gradient_episode_summary(prefix, evaluations):
    """Read scalar norms through the participant API when tensor support exists.

    The evaluation-loss fallback remains prefix-conditioned and makes the
    interface executable in a minimal numerical runtime.  It is conservative:
    it supplies only the most recent observed shock and the structural recharge
    scale; it does not contain a future calendar.
    """
    scalar_rows = []
    try:
        objects = prefix.objects(role="gradient_total_norm", materialized=True)
        objects.sort(key=lambda row: int(row["optimizer_step"]))
        if objects:
            probe = prefix.load_tensor(objects[-1])
            float(probe.item() if hasattr(probe, "item") else probe)
            for row in objects:
                value = prefix.load_tensor(row)
                value = float(value.item() if hasattr(value, "item") else value)
                scalar_rows.append((int(row["optimizer_step"]), value))
    except (AttributeError, ImportError, ModuleNotFoundError, RuntimeError,
            TypeError, ValueError):
        scalar_rows = []

    onsets = []
    peaks = []
    if scalar_rows:
        groups = []
        for step, value in scalar_rows:
            if value <= 1.0:
                continue
            if not groups or step - groups[-1][-1][0] > 50:
                groups.append([])
            groups[-1].append((step, value))
        for group in groups:
            if group:
                onsets.append(group[0][0])
                peaks.append(max(value for _, value in group))

    if onsets:
        current_step = int(evaluations[-1]["optimizer_step"])
        last_onset = int(onsets[-1])
        intervals = [b - a for a, b in zip(onsets, onsets[1:])]
        recent = intervals[-4:]
        recharge = max(330.0, min(370.0, _median(recent, 330.0) + 20.0))
        peak = float(peaks[-1]) if peaks else 1.0
        return {
            "source": "materialized_gradient_total_norm",
            "last_onset": last_onset,
            "phase": max(0.0, current_step - last_onset),
            "recharge_period": recharge,
            "recent_intervals": [float(value) for value in recent],
            "last_peak": peak,
        }

    shocks = [
        int(row["optimizer_step"])
        for row in evaluations
        if int(row["optimizer_step"]) >= 100 and float(row["loss"]) > 0.05
    ]
    current_step = int(evaluations[-1]["optimizer_step"])
    if shocks:
        # Evaluations are on a 100-step grid; the observed discovery shocks
        # begin near the preceding ten optimizer updates.
        last_onset = max(0, shocks[-1] - 10)
    else:
        last_onset = max(0, current_step - 300)
    return {
        "source": "evaluation_shock_fallback",
        "last_onset": int(last_onset),
        "phase": max(0.0, current_step - last_onset),
        "recharge_period": 350.0,
        "recent_intervals": [],
        "last_peak": 1.0,
    }


class FormationDynamics:
    """Rule formation from validation charge and clipped-episode impulses."""

    def initialize(self, gfg_prefix):
        evaluations = _rows(gfg_prefix)
        if not evaluations:
            raise ValueError("formation initialization requires an evaluation prefix")
        last = evaluations[-1]
        observed_capability = max(
            0.0, min(1.0, float(last["validation_accuracy"]))
        )
        observed_transition = _observed_transition(evaluations)
        if observed_transition is None:
            capability = observed_capability
        else:
            capability = max(
                0.90,
                max(
                    float(row["validation_accuracy"])
                    for row in evaluations
                    if int(row["optimizer_step"]) >= observed_transition
                ),
            )
        recent = evaluations[-3:]
        positive_change = 0.0
        elapsed = 0
        for left, right in zip(recent, recent[1:]):
            delta_step = int(right["optimizer_step"]) - int(left["optimizer_step"])
            if delta_step > 0:
                positive_change += max(
                    0.0,
                    float(right["validation_accuracy"])
                    - float(left["validation_accuracy"]),
                )
                elapsed += delta_step
        velocity = positive_change / elapsed if elapsed else 0.0001
        velocity = max(0.00008, min(0.00012, velocity))
        sustained = observed_transition is not None
        return {
            "step": int(last["optimizer_step"]),
            "capability": capability,
            "formation_velocity": velocity,
            "formed": bool(sustained),
            "transition_step": observed_transition,
            "last_observed_validation": observed_capability,
            "last_observed_train": float(last["train_accuracy"]),
            "episode_impulses_used": 0,
            "impulse_scale": 1.0,
            "operative_fields": [
                "capability",
                "formation_velocity",
                "formed",
                "episode_impulses_used",
                "impulse_scale",
            ],
        }

    def step(self, state, context):
        next_state = dict(state)
        target_step = int(context["step"])
        delta = max(0, target_step - int(state["step"]))
        capability = float(state["capability"])
        if bool(state["formed"]):
            capability += (1.0 - capability) * (
                1.0 - math.exp(-0.0025 * delta)
            )
        else:
            capability += float(state["formation_velocity"]) * delta

        impulses = int(context.get("episode_impulse", 0))
        for _ in range(impulses):
            if capability < 0.20:
                gain = 0.08
            elif capability < 0.60:
                gain = 0.72
            elif capability < 0.90:
                gain = 0.16
            else:
                gain = 0.55
            capability += (
                gain * float(state.get("impulse_scale", 1.0))
                * (1.0 - capability)
            )

        capability = max(0.0, min(1.0, capability))
        formed = bool(state["formed"]) or capability >= 0.90
        transition_step = state.get("transition_step")
        if formed and transition_step is None:
            transition_step = target_step
        if formed:
            capability = max(0.90, capability)

        next_state.update({
            "step": target_step,
            "capability": capability,
            "formed": formed,
            "transition_step": transition_step,
            "episode_impulses_used": int(state["episode_impulses_used"])
            + impulses,
        })
        return next_state

    def output(self, state):
        return max(0.0, min(1.0, float(state["capability"])))


class StabilityDynamics:
    """Clipped-impulse recharge, burden, degradation, and recovery dynamics."""

    def initialize(self, gfg_prefix):
        evaluations = _rows(gfg_prefix)
        if not evaluations:
            raise ValueError("stability initialization requires an evaluation prefix")
        summary = _gradient_episode_summary(gfg_prefix, evaluations)
        current_step = int(evaluations[-1]["optimizer_step"])
        observed_transition = _observed_transition(evaluations)
        severe_steps = []
        if observed_transition is not None:
            severe_steps = [
                int(row["optimizer_step"])
                for row in evaluations
                if int(row["optimizer_step"]) > observed_transition
                and float(row["validation_accuracy"]) < 0.90
            ]

        period = float(summary["recharge_period"])
        last_onset = float(summary["last_onset"])
        burden = 0.0
        recovery_memory = 0
        fragility_done = False
        if observed_transition is not None:
            if severe_steps:
                fragility_done = True
                recovery_memory = max(0, len(severe_steps) - 1)
                last_onset = float(severe_steps[-1])
                if recovery_memory:
                    period = 395.0
                    burden = 0.72
                else:
                    period = 358.0
                    burden = 0.21
                next_episode = last_onset + period
                while next_episode <= current_step:
                    last_onset = next_episode
                    burden += 0.07
                    next_episode += period
            else:
                period = 380.0
                next_episode = float(observed_transition + 40)
                last_onset = next_episode - period
                while next_episode <= current_step:
                    last_onset = next_episode
                    burden += 0.07
                    next_episode += period
        else:
            next_episode = last_onset + period
            while next_episode <= current_step:
                last_onset = next_episode
                next_episode += period
        return {
            "step": current_step,
            "last_episode_onset": last_onset,
            "next_episode_onset": next_episode,
            "recharge_period": period,
            "phase": max(0.0, current_step - last_onset),
            "impulse_burden": burden,
            "impulse_scale": 1.0,
            "episode_energy": max(1.0, min(2.0, float(summary["last_peak"]) / 50.0)),
            "recovery_memory": recovery_memory,
            "fragility_discharge_done": fragility_done,
            "formed": observed_transition is not None,
            "formation_step": observed_transition,
            "shock_onset": None,
            "shock_amplitude": 0.0,
            "shock_kind": "NONE",
            "degradation": 0.0,
            "last_episode_count": 0,
            "evidence_source": summary["source"],
            "recent_recharge_intervals": summary["recent_intervals"],
            "operative_fields": [
                "next_episode_onset",
                "recharge_period",
                "impulse_burden",
                "impulse_scale",
                "recovery_memory",
                "fragility_discharge_done",
                "shock_onset",
                "shock_amplitude",
            ],
        }

    def step(self, state, context):
        next_state = dict(state)
        target_step = int(context["step"])
        formation_capability = float(context.get("formation_capability", 0.0))
        transition_step = context.get("formation_transition_step")
        if not bool(next_state["formed"]) and formation_capability >= 0.90:
            next_state["formed"] = True
            next_state["formation_step"] = int(
                transition_step if transition_step is not None else state["step"]
            )
            next_state["impulse_burden"] = 0.0

        episode_count = 0
        next_onset = float(next_state["next_episode_onset"])
        while next_onset <= target_step + 1e-9:
            onset = next_onset
            episode_count += 1
            next_state["last_episode_onset"] = onset
            next_state["episode_energy"] = min(
                2.0, 0.96 * float(next_state["episode_energy"]) + 0.08
            )
            if bool(next_state["formed"]):
                impulse_scale = float(next_state.get("impulse_scale", 1.0))
                burden = (
                    float(next_state["impulse_burden"])
                    + 0.07 * impulse_scale
                )
                next_state["impulse_burden"] = burden
                age = onset - float(next_state["formation_step"])
                fragile = (
                    not bool(next_state["fragility_discharge_done"])
                    and burden >= 0.20
                    and 600.0 <= age <= 1200.0
                )
                accumulated = burden >= 0.995
                if fragile:
                    next_state["shock_onset"] = onset
                    next_state["shock_amplitude"] = 0.44
                    next_state["shock_kind"] = "FORMATION_FRAGILITY"
                    next_state["fragility_discharge_done"] = True
                elif accumulated:
                    memory = int(next_state["recovery_memory"])
                    next_state["shock_onset"] = onset
                    next_state["shock_amplitude"] = 0.81 if memory == 0 else 0.31
                    next_state["shock_kind"] = "ACCUMULATED_IMPULSE_BURDEN"
                    next_state["recovery_memory"] = memory + 1
                    next_state["impulse_burden"] = 0.72

                if accumulated or int(next_state["recovery_memory"]) > 0:
                    period = 395.0
                elif age < 600.0:
                    period = 380.0
                else:
                    period = 358.0
            else:
                period = 350.0
            next_state["recharge_period"] = period
            next_onset = onset + period

        next_state["next_episode_onset"] = next_onset
        next_state["last_episode_count"] = episode_count
        next_state["step"] = target_step
        next_state["phase"] = max(
            0.0, target_step - float(next_state["last_episode_onset"])
        )

        degradation = 0.0
        if bool(next_state["formed"]):
            ordinary_age = target_step - float(next_state["last_episode_onset"])
            if 0.0 <= ordinary_age <= 90.0:
                degradation = 0.035 * math.exp(-ordinary_age / 30.0)
            shock_onset = next_state.get("shock_onset")
            if shock_onset is not None:
                shock_age = target_step - float(shock_onset)
                if 0.0 <= shock_age <= 90.0:
                    shock = float(next_state["shock_amplitude"]) * math.exp(
                        -shock_age / 70.0
                    )
                    degradation = max(degradation, shock)
        next_state["degradation"] = max(0.0, min(1.0, degradation))
        return next_state

    def output(self, state):
        return max(0.0, min(1.0, float(state["degradation"])))


class CapabilityDynamicsMechanism:
    """Compose formation capability and stability degradation exactly."""

    def __init__(self):
        self.formation = FormationDynamics()
        self.stability = StabilityDynamics()

    def initialize(self, gfg_prefix):
        formation_state = self.formation.initialize(gfg_prefix)
        stability_state = self.stability.initialize(gfg_prefix)
        return {
            "formation_state": formation_state,
            "stability_state": stability_state,
        }

    def forecast(self, state):
        formation_state = dict(state["formation_state"])
        stability_state = dict(state["stability_state"])
        cut = max(int(formation_state["step"]), int(stability_state["step"]))
        first_grid = ((cut // 100) + 1) * 100
        grid = list(range(first_grid, 10001, 100))

        formation_curve = []
        stability_curve = []
        validation_curve = []
        trajectory = []

        for step in grid:
            stability_state = self.stability.step(stability_state, {
                "step": step,
                "formation_capability": self.formation.output(formation_state),
                "formation_transition_step": formation_state.get("transition_step"),
            })
            formation_state = self.formation.step(formation_state, {
                "step": step,
                "episode_impulse": stability_state.get("last_episode_count", 0),
            })
            capability = self.formation.output(formation_state)
            degradation = self.stability.output(stability_state)
            accuracy = max(0.0, min(1.0, capability - degradation))
            formation_curve.append({"step": step, "capability": capability})
            stability_curve.append({"step": step, "degradation": degradation})
            validation_curve.append({"step": step, "accuracy": accuracy})
            trajectory.append({
                "step": step,
                "formation_state": dict(formation_state),
                "stability_state": dict(stability_state),
            })

        transition = state["formation_state"].get("transition_step")
        if transition is not None:
            transition = int(transition)
        else:
            for index in range(max(0, len(validation_curve) - 2)):
                window = validation_curve[index:index + 3]
                if len(window) == 3 and all(
                    row["accuracy"] >= 0.90 for row in window
                ):
                    transition = int(window[0]["step"])
                    break
        will_transition = transition is not None and transition <= 10000

        event_steps = []
        if transition is not None:
            event_steps = [
                int(row["step"])
                for row in validation_curve
                if int(row["step"]) > transition and row["accuracy"] < 0.90
            ]
        intervals = []
        for step in event_steps:
            if intervals and step == intervals[-1]["step_high"] + 100:
                intervals[-1]["step_high"] = step
            else:
                intervals.append({"step_low": step, "step_high": step})

        if event_steps:
            recovered = any(
                row["step"] > event_steps[-1] and row["accuracy"] >= 0.90
                for row in validation_curve
            )
            stability_class = (
                "TRANSIENT_DEGRADATION_RECOVERY"
                if recovered else "PERSISTENT_DEGRADATION"
            )
        elif transition is not None:
            stability_class = "STABLE"
        else:
            stability_class = "UNDETERMINED"

        if transition is None:
            low_200 = high_200 = "NO_TRANSITION"
            low_500 = high_500 = "NO_TRANSITION"
        else:
            low_200 = max(0, transition - 100)
            high_200 = min(10000, low_200 + 200)
            low_500 = max(0, transition - 200)
            high_500 = min(10000, low_500 + 400)

        return {
            "will_transition": will_transition,
            "transition_step_low_200": low_200,
            "transition_step_high_200": high_200,
            "transition_step_low_500": low_500,
            "transition_step_high_500": high_500,
            "predicted_formation_curve": formation_curve,
            "predicted_stability_degradation_curve": stability_curve,
            "predicted_validation_curve": validation_curve,
            "mechanism_state": {
                "formation_state": formation_state,
                "stability_state": stability_state,
                "state_trajectory": trajectory,
            },
            "post_formation_stability": stability_class,
            "predicted_instability_intervals": intervals,
        }
