from pathlib import Path

from experiments.w3c_prov_projection_v1.src import authority_store_audit as authority
from experiments.w3c_prov_projection_v1.src import oracle_isolation_audit as oracle


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_authority_store_scan_is_rule_driven_and_closed() -> None:
    policy = authority.load_policy(EXPERIMENT_ROOT)
    scan = authority.scan_repository(EXPERIMENT_ROOT, policy)
    assert scan["status"] == "PASS"
    assert scan["scanned_file_count"] == scan["classified_file_count"]
    assert scan["unclassified_file_count"] == 0
    assert scan["forbidden_secondary_relation_store_count"] == 0
    assert scan["persistent_candidate_lookup_table_count"] == 0
    assert scan["hidden_binding_crosswalk_count"] == 0
    assert scan["snapshot_blob_embedded_in_prov_count"] == 0


def test_authority_store_mutations_fail_closed() -> None:
    result = authority.run_authority_negative_controls(authority.load_policy(EXPERIMENT_ROOT))
    assert result["status"] == "SUPPORTED"
    assert result["detected_count"] == result["negative_control_count"] == 10
    assert all(item["status"] == "FAIL_CLOSED" for item in result["controls"])


def test_oracle_import_graph_has_only_neutral_sharing() -> None:
    policy = oracle.load_policy(EXPERIMENT_ROOT)
    graph = oracle.analyze_import_graph(EXPERIMENT_ROOT / "src", policy)
    assert graph["status"] == "PASS"
    assert not graph["candidate_imports_native"]
    assert not graph["native_imports_candidate"]
    assert not graph["native_imports_core"]
    assert not graph["candidate_uses_provo_normalizer"]
    assert not graph["native_uses_provn_parser"]
    assert graph["shared_neutral_modules"] == ["record_model"]
    assert graph["shared_mapping_helper_count"] == 0


def test_oracles_execute_in_independent_traced_processes() -> None:
    oracle_policy = oracle.load_policy(EXPERIMENT_ROOT)
    graph = oracle.analyze_import_graph(EXPERIMENT_ROOT / "src", oracle_policy)
    process_trace = oracle.run_oracle_process_audit(EXPERIMENT_ROOT, oracle_policy)
    isolation = oracle.build_oracle_isolation(graph, process_trace)
    runtime = authority.build_runtime_authority_trace(process_trace)
    second = authority.compute_second_authority_audit(
        authority.scan_repository(EXPERIMENT_ROOT, authority.load_policy(EXPERIMENT_ROOT)),
        runtime,
        authority.load_policy(EXPERIMENT_ROOT),
    )
    assert process_trace["status"] == "PASS"
    assert process_trace["run_count_per_path"] == 2
    assert len(process_trace["runs"]) == 4
    assert all(item["process_boundary"] == "separate child process" for item in process_trace["runs"])
    assert isolation["status"] == "SUPPORTED"
    assert isolation["normalized_record_count"] == 51
    assert not isolation["process_memory_shared"]
    assert runtime["status"] == "PASS"
    assert runtime["forbidden_read_count"] == 0
    assert second["status"] == "SUPPORTED"
    assert second["second_authority_count"] == 0


def test_oracle_isolation_mutations_fail_closed() -> None:
    result = oracle.run_oracle_negative_controls(EXPERIMENT_ROOT, oracle.load_policy(EXPERIMENT_ROOT))
    assert result["status"] == "SUPPORTED"
    assert result["detected_count"] == result["negative_control_count"] == 10
    assert all(item["status"] == "FAIL_CLOSED" for item in result["controls"])
