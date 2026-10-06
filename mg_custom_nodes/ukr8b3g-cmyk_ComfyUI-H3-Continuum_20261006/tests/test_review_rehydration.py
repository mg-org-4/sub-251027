"""Exercise real frontend lifecycle hooks with recreated workflow nodes."""


def test_review_workflow_rehydration(review_queue_results):
    cases = (
        "H3ContinuumSamplerV38: recreated tab waits for saved identity and upstream links",
        "H3ContinuumSamplerV39: recreated tab waits for saved identity and upstream links",
        "ten rapid recreated tabs reject every obsolete history response",
        "API graph restoration without a before hook still reloads history",
        "manually added sampler loads history after deferred setup",
        "real edits while restored history is pending retain the settings guard",
        "same revision graph refresh cannot erase an existing real edit",
        "graph epoch rejects old results even if the same node remains attached",
        "removed sampler cancels deferred setup without fetching or mutating it",
        "initial restored history read preserves saved Take selection fields",
        "restored missing or failed history reports accurately and can retry",
        "failed graph load releases hydration gate for subsequent manual nodes",
    )
    for case in cases:
        assert review_queue_results[case]["pass"]
