from typing import Dict, List, Tuple

HAND_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === ego4d + egoexo4d + ssv2 + epic Dataset ===
    "magic_mix": [
        ("ego4d_cooking_and_cleaning", 1.0),
        ("egoexo4d", 3.0),
        ("epic", 1.0),
        ('ssv2', 5.0),
        ('ego4d_other', 0.5)
    ],
    "magic_mix_cooking_and_cleaning": [
        ("ego4d_cooking_and_cleaning", 1.0),
        ("egoexo4d", 3.0),
        ("epic", 1.0),
        ('ssv2', 5.0),
    ],
    "gigahands_demo_only": [
        ("gigahands", 1.0),
    ],
    "gigahands_real_train": [
        ("gigahands_real_train", 1.0),
    ],
    "gigahands_real_test": [
        ("gigahands_real_test", 1.0),
    ],
    "opentouch_keypoint_train": [
        ("opentouch_keypoint_train", 1.0),
    ],
    "opentouch_keypoint_test": [
        ("opentouch_keypoint_test", 1.0),
    ],
    "magic_mix_plus_gigahands": [
        ("ego4d_cooking_and_cleaning", 1.0),
        ("egoexo4d", 3.0),
        ("epic", 1.0),
        ('ssv2', 5.0),
        ('ego4d_other', 0.5),
        ("gigahands", 1.0),
    ],
}
