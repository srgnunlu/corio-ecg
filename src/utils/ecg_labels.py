# ECGFounder 150-class diagnosis labels and related constants
# Used by the diagnosis module to map model output indices to human-readable labels

LEAD_NAMES: list[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

NUM_CLASSES: int = 150

DEFAULT_THRESHOLD: float = 0.5

# Indices of clinically critical diagnoses that require immediate attention
CRITICAL_DIAGNOSIS_INDICES: list[int] = [
    5,    # ATRIAL FIBRILLATION
    87,   # ACUTE MI / STEMI
    93,   # SUPRAVENTRICULAR TACHYCARDIA
    95,   # WIDE QRS TACHYCARDIA
    98,   # VENTRICULAR TACHYCARDIA
    121,  # WITH COMPLETE HEART BLOCK
    130,  # ACUTE MI
    131,  # ACUTE PERICARDITIS
]

# All 150 ECGFounder diagnosis labels in order (0-indexed)
ECG_FOUNDER_LABELS: list[str] = [
    "ABNORMAL ECG",                                                         # 0
    "NORMAL SINUS RHYTHM",                                                  # 1
    "NORMAL ECG",                                                           # 2
    "SINUS RHYTHM",                                                         # 3
    "SINUS BRADYCARDIA",                                                    # 4
    "ATRIAL FIBRILLATION",                                                  # 5
    "SINUS TACHYCARDIA",                                                    # 6
    "OTHERWISE NORMAL ECG",                                                 # 7
    "LEFT AXIS DEVIATION",                                                  # 8
    "PREMATURE VENTRICULAR COMPLEXES",                                      # 9
    "BORDERLINE ECG",                                                       # 10
    "RIGHT BUNDLE BRANCH BLOCK",                                            # 11
    "SEPTAL INFARCT",                                                       # 12
    "LEFT ATRIAL ENLARGEMENT",                                              # 13
    "NONSPECIFIC T WAVE ABNORMALITY",                                       # 14
    "LOW VOLTAGE QRS",                                                      # 15
    "PREMATURE ATRIAL COMPLEXES",                                           # 16
    "ANTERIOR INFARCT",                                                     # 17
    "INCOMPLETE RIGHT BUNDLE BRANCH BLOCK",                                 # 18
    "PREMATURE SUPRAVENTRICULAR COMPLEXES",                                 # 19
    "LEFT BUNDLE BRANCH BLOCK",                                             # 20
    "NONSPECIFIC T WAVE ABNORMALITY NOW EVIDENT IN",                        # 21
    "NONSPECIFIC T WAVE ABNORMALITY NO LONGER EVIDENT IN",                  # 22
    "T WAVE INVERSION NOW EVIDENT IN",                                      # 23
    "LATERAL INFARCT",                                                      # 24
    "NONSPECIFIC ST ABNORMALITY",                                           # 25
    "LEFT VENTRICULAR HYPERTROPHY",                                         # 26
    "T WAVE INVERSION NO LONGER EVIDENT IN",                                # 27
    "WITH RAPID VENTRICULAR RESPONSE",                                      # 28
    "QT HAS SHORTENED",                                                     # 29
    "QT HAS LENGTHENED",                                                    # 30
    "FUSION COMPLEXES",                                                     # 31
    "ATRIAL FLUTTER",                                                       # 32
    "MARKED SINUS BRADYCARDIA",                                             # 33
    "WITH SINUS ARRHYTHMIA",                                                # 34
    "NONSPECIFIC ST AND T WAVE ABNORMALITY",                                # 35
    "LEFT ANTERIOR FASCICULAR BLOCK",                                       # 36
    "RIGHT AXIS DEVIATION",                                                 # 37
    "ECTOPIC ATRIAL RHYTHM",                                                # 38
    "UNDETERMINED RHYTHM",                                                  # 39
    "ANTEROSEPTAL INFARCT",                                                 # 40
    "RIGHTWARD AXIS",                                                       # 41
    "ST NOW DEPRESSED IN",                                                  # 42
    "WITH SHORT PR",                                                        # 43
    "WITH MARKED SINUS ARRHYTHMIA",                                        # 44
    "ST NO LONGER DEPRESSED IN",                                            # 45
    "INVERTED T WAVES HAVE REPLACED NONSPECIFIC T WAVE ABNORMALITY IN",    # 46
    "NON-SPECIFIC CHANGE IN ST SEGMENT IN",                                 # 47
    "NONSPECIFIC T WAVE ABNORMALITY HAS REPLACED INVERTED T WAVES IN",     # 48
    "JUNCTIONAL RHYTHM",                                                    # 49
    "ELECTRONIC ATRIAL PACEMAKER",                                          # 50
    "ABERRANT CONDUCTION",                                                  # 51
    "ELECTRONIC VENTRICULAR PACEMAKER",                                     # 52
    "T WAVE INVERSION LESS EVIDENT IN",                                     # 53
    "ANTEROLATERAL INFARCT",                                                # 54
    "WITH REPOLARIZATION ABNORMALITY",                                      # 55
    "RSR' OR QR PATTERN IN V1 SUGGESTS RIGHT VENTRICULAR CONDUCTION DELAY", # 56
    "T WAVE INVERSION MORE EVIDENT IN",                                     # 57
    "WIDE QRS RHYTHM",                                                      # 58
    "WITH PREMATURE VENTRICULAR OR ABERRANTLY CONDUCTED COMPLEXES",         # 59
    "RIGHT ATRIAL ENLARGEMENT",                                             # 60
    "INFERIOR INFARCT",                                                     # 61
    "INCOMPLETE LEFT BUNDLE BRANCH BLOCK",                                  # 62
    "VOLTAGE CRITERIA FOR LEFT VENTRICULAR HYPERTROPHY",                    # 63
    "OR DIGITALIS EFFECT",                                                  # 64
    "BIFASCICULAR BLOCK",                                                   # 65
    "ST NO LONGER ELEVATED IN",                                             # 66
    "WITH SLOW VENTRICULAR RESPONSE",                                       # 67
    "ST ELEVATION NOW PRESENT IN",                                          # 68
    "PREMATURE ECTOPIC COMPLEXES",                                          # 69
    "LEFT POSTERIOR FASCICULAR BLOCK",                                      # 70
    "T WAVE AMPLITUDE HAS DECREASED IN",                                    # 71
    "WITH A COMPETING JUNCTIONAL PACEMAKER",                                # 72
    "RIGHT SUPERIOR AXIS DEVIATION",                                        # 73
    "BIATRIAL ENLARGEMENT",                                                 # 74
    "VENTRICULAR-PACED RHYTHM",                                             # 75
    "ATRIAL-PACED RHYTHM",                                                  # 76
    "T WAVE AMPLITUDE HAS INCREASED IN",                                    # 77
    "WITH QRS WIDENING",                                                    # 78
    "WITH 1ST DEGREE AV BLOCK",                                             # 79
    "PROLONGED QT",                                                         # 80
    "WITH PROLONGED AV CONDUCTION",                                         # 81
    "RIGHT VENTRICULAR HYPERTROPHY",                                        # 82
    "WITH QRS WIDENING AND REPOLARIZATION ABNORMALITY",                     # 83
    "ATRIAL-SENSED VENTRICULAR-PACED RHYTHM",                               # 84
    "AV SEQUENTIAL OR DUAL CHAMBER ELECTRONIC PACEMAKER",                   # 85
    "PULMONARY DISEASE PATTERN",                                            # 86
    "ACUTE MI / STEMI",                                                     # 87
    "INFERIOR-POSTERIOR INFARCT",                                           # 88
    "NONSPECIFIC INTRAVENTRICULAR CONDUCTION DELAY",                        # 89
    "PREMATURE VENTRICULAR AND FUSION COMPLEXES",                           # 90
    "IN A PATTERN OF BIGEMINY",                                             # 91
    "AV DUAL-PACED RHYTHM",                                                 # 92
    "SUPRAVENTRICULAR TACHYCARDIA",                                         # 93
    "VENTRICULAR-PACED COMPLEXES",                                          # 94
    "WIDE QRS TACHYCARDIA",                                                 # 95
    "RSR' PATTERN IN V1",                                                   # 96
    "ST LESS DEPRESSED IN",                                                 # 97
    "VENTRICULAR TACHYCARDIA",                                              # 98
    "EARLY REPOLARIZATION",                                                 # 99
    "ST MORE DEPRESSED IN",                                                 # 100
    "ANTEROLATERAL LEADS",                                                  # 101
    "ELECTRONIC DEMAND PACING",                                             # 102
    "RBBB AND LEFT ANTERIOR FASCICULAR BLOCK",                              # 103
    "LATERAL INJURY PATTERN",                                               # 104
    "BIVENTRICULAR PACEMAKER DETECTED",                                     # 105
    "SUSPECT UNSPECIFIED PACEMAKER FAILURE",                                 # 106
    "WOLFF-PARKINSON-WHITE",                                                # 107
    "WITH VENTRICULAR ESCAPE COMPLEXES",                                    # 108
    "INFERIOR INJURY PATTERN",                                              # 109
    "CONSIDER RIGHT VENTRICULAR INVOLVEMENT IN ACUTE INFERIOR INFARCT",     # 110
    "ST ELEVATION HAS REPLACED ST DEPRESSION IN",                           # 111
    "NONSPECIFIC INTRAVENTRICULAR BLOCK",                                    # 112
    "MASKED BY FASCICULAR BLOCK",                                           # 113
    "PEDIATRIC ECG ANALYSIS",                                               # 114
    "BLOCKED",                                                              # 115
    "WITH UNDETERMINED RHYTHM IRREGULARITY",                                # 116
    "LEFTWARD AXIS",                                                        # 117
    "WITH 2ND DEGREE SA BLOCK MOBITZ I",                                    # 118
    "ACUTE",                                                                # 119
    "ABNORMAL LEFT AXIS DEVIATION",                                         # 120
    "WITH COMPLETE HEART BLOCK",                                            # 121
    "NO P-WAVES FOUND",                                                     # 122
    "ST LESS ELEVATED IN",                                                  # 123
    "WITH RETROGRADE CONDUCTION",                                           # 124
    "ST MORE ELEVATED IN",                                                  # 125
    "JUNCTIONAL BRADYCARDIA",                                               # 126
    "WITH VARIABLE AV BLOCK",                                               # 127
    "ANTERIOR INJURY PATTERN",                                              # 128
    "WITH JUNCTIONAL ESCAPE COMPLEXES",                                     # 129
    "ACUTE MI",                                                             # 130
    "ACUTE PERICARDITIS",                                                   # 131
    "POSTERIOR INFARCT",                                                     # 132
    "IDIOVENTRICULAR RHYTHM",                                               # 133
    "WITH 2ND DEGREE SA BLOCK MOBITZ II",                                   # 134
    "R IN AVL",                                                             # 135
    "SINUS/ATRIAL CAPTURE",                                                 # 136
    "AV DUAL-PACED COMPLEXES",                                              # 137
    "INFEROLATERAL INJURY PATTERN",                                         # 138
    "RBBB AND LEFT POSTERIOR FASCICULAR BLOCK",                              # 139
    "ANTEROLATERAL INJURY PATTERN",                                         # 140
    "ATRIAL-PACED COMPLEXES",                                               # 141
    "WITH SINUS PAUSE",                                                     # 142
    "BIVENTRICULAR HYPERTROPHY",                                            # 143
    "ABNORMAL RIGHT AXIS DEVIATION",                                        # 144
    "SUPRAVENTRICULAR COMPLEXES",                                           # 145
    "WITH 2ND DEGREE AV BLOCK MOBITZ I",                                    # 146
    "WITH 2:1 AV CONDUCTION",                                               # 147
    "WITH AV DISSOCIATION",                                                 # 148
    "MULTIFOCAL ATRIAL TACHYCARDIA",                                        # 149
]
