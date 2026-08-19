from collections import Counter


def test_native_endpoint_signatures_are_preserved(order_run):
    result, _ = order_run
    assert result["compiled_primitive_relation_count"] == 83
    assert result["endpoint_signature_counts"] == {
        "generation_fact->generation_fact": 23,
        "generation_occurrence->generation_occurrence": 60,
    }
    assert Counter(result["relation_type_counts"]) == Counter(
        {
            "commits_version": 4,
            "conflicts_with": 2,
            "generated_origin_dependency": 10,
            "message_send_receive": 5,
            "program_order": 46,
            "reads_from": 7,
            "synchronizes_with": 9,
        }
    )
