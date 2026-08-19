from __future__ import annotations

from typing import Any

from rdflib import Graph, Literal, Namespace, RDF, URIRef

from .record_model import sorted_records


EX = Namespace("https://example.org/w3c-prov-projection-v1#")
PROV = Namespace("http://www.w3.org/ns/prov#")


_ATTRIBUTE_NAMES = [
    "sourceIdentity", "sourceGranularity", "domainType", "stableDomainIdentity",
    "resultCategory", "resultIdentity", "dispositionCategory", "reasonCode",
    "occurrenceStage", "occurrenceType", "stableInstanceKey", "occurrenceIndex",
    "operationType", "generatorName", "generatorVersion", "codeIdentity",
]


def _qname(value: URIRef) -> str:
    text = str(value)
    if text.startswith(str(EX)):
        return "ex:" + text[len(str(EX)):]
    if text.startswith(str(PROV)):
        return "prov:" + text[len(str(PROV)):]
    raise ValueError(f"URI outside frozen namespaces: {text}")


def _one(graph: Graph, subject: URIRef, predicate: URIRef) -> Any:
    values = list(graph.objects(subject, predicate))
    if len(values) != 1:
        raise ValueError(f"qualified PROV-O cardinality error: {_qname(subject)} {_qname(predicate)} count={len(values)}")
    return values[0]


def _owner(graph: Graph, predicate: URIRef, node: URIRef) -> URIRef:
    values = list(graph.subjects(predicate, node))
    if len(values) != 1 or not isinstance(values[0], URIRef):
        raise ValueError(f"qualified PROV-O owner error: {_qname(node)}")
    return values[0]


def _node_record(graph: Graph, node: URIRef, kind: str, base_type: URIRef) -> dict[str, Any]:
    types = sorted(_qname(value) for value in graph.objects(node, RDF.type) if value != base_type and value != PROV.Agent)
    attributes: dict[str, Any] = {}
    for name in _ATTRIBUTE_NAMES:
        values = list(graph.objects(node, EX[name]))
        if len(values) > 1:
            raise ValueError(f"duplicate PROV-O attribute: {_qname(node)} ex:{name}")
        if values:
            value = values[0]
            attributes["ex:" + name] = value.toPython() if isinstance(value, Literal) else _qname(value)
    return {"kind": kind, "id": _qname(node), "types": types, "attributes": dict(sorted(attributes.items()))}


def normalize_provo(document: bytes) -> list[dict[str, Any]]:
    graph = Graph()
    graph.parse(data=document.decode("utf-8"), format="turtle", publicID="urn:w3c-prov-projection-v1")
    records: list[dict[str, Any]] = []
    influence_types = {PROV.Usage, PROV.Generation, PROV.Derivation, PROV.Association}
    for node in sorted({value for value in graph.subjects(RDF.type, PROV.Entity)}, key=str):
        if not isinstance(node, URIRef):
            raise ValueError("blank Entity is prohibited")
        records.append(_node_record(graph, node, "entity", PROV.Entity))
    for node in sorted({value for value in graph.subjects(RDF.type, PROV.Activity)}, key=str):
        if not isinstance(node, URIRef):
            raise ValueError("blank Activity is prohibited")
        records.append(_node_record(graph, node, "activity", PROV.Activity))
    agents = set(graph.subjects(RDF.type, PROV.Agent)) | set(graph.subjects(RDF.type, PROV.SoftwareAgent))
    for node in sorted(agents, key=str):
        if not isinstance(node, URIRef):
            raise ValueError("blank Agent is prohibited")
        records.append(_node_record(graph, node, "agent", PROV.Agent))
    for node in sorted({value for value in graph.subjects(RDF.type, PROV.Usage)}, key=str):
        activity = _owner(graph, PROV.qualifiedUsage, node)
        entity = _one(graph, node, PROV.entity)
        role = _one(graph, node, PROV.hadRole)
        ordinal = _one(graph, node, EX.relationOrdinal)
        records.append({"kind": "usage", "id": _qname(node), "activity": _qname(activity), "entity": _qname(entity), "role": _qname(role), "ordinal": int(ordinal)})
    for node in sorted({value for value in graph.subjects(RDF.type, PROV.Generation)}, key=str):
        entity = _owner(graph, PROV.qualifiedGeneration, node)
        activity = _one(graph, node, PROV.activity)
        records.append({"kind": "generation", "id": _qname(node), "entity": _qname(entity), "activity": _qname(activity)})
    for node in sorted({value for value in graph.subjects(RDF.type, PROV.Derivation)}, key=str):
        generated = _owner(graph, PROV.qualifiedDerivation, node)
        records.append({
            "kind": "derivation", "id": _qname(node), "generated_entity": _qname(generated),
            "used_entity": _qname(_one(graph, node, PROV.entity)),
            "activity": _qname(_one(graph, node, PROV.hadActivity)),
            "generation": _qname(_one(graph, node, PROV.hadGeneration)),
            "usage": _qname(_one(graph, node, PROV.hadUsage)),
            "role": _qname(_one(graph, node, PROV.hadRole)),
            "ordinal": int(_one(graph, node, EX.relationOrdinal)),
        })
    for node in sorted({value for value in graph.subjects(RDF.type, PROV.Association)}, key=str):
        activity = _owner(graph, PROV.qualifiedAssociation, node)
        records.append({
            "kind": "association", "id": _qname(node), "activity": _qname(activity),
            "agent": _qname(_one(graph, node, PROV.agent)),
            "role": _qname(_one(graph, node, PROV.hadRole)),
            "ordinal": int(_one(graph, node, EX.relationOrdinal)),
        })
    typed_influences = {
        node for relation_type in influence_types for node in graph.subjects(RDF.type, relation_type)
    }
    normalized_influences = {record["id"] for record in records if record["kind"] in {"usage", "generation", "derivation", "association"}}
    if {_qname(node) for node in typed_influences} != normalized_influences:
        raise ValueError("unparsed qualified PROV-O influence")
    return sorted_records(records)

