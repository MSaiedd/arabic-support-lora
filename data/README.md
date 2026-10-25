# Original synthetic support fixture

32 train / 16 validation / 24 test examples. Four categories and two urgency labels.
English and Arabic occur in every split; mixed-language tickets are test-only.
IDs and normalized text must be disjoint across all splits.

`scripts/create_fixture.py` is the reproducible source. Text and responses are original
synthetic examples licensed under the repository's MIT license. No external dataset was copied.
Urgency high means explicit account compromise, unauthorized/duplicate payment, an expiring
or overdue refund requiring urgent attention, or a missing/stolen delivery needing escalation.
These conventions are examples, not a universal support SLA.

The data is a smoke fixture. Replace/expand it with reviewed consented examples, scenario-based
splits, semantic duplicate checks, and independent test labels for meaningful model assessment.
