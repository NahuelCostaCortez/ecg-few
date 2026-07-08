Analyze this single right precordial ECG beat image from lead {lead} from one patient.

Decide whether the patient is compatible with the Brugada-positive research
reference used in this study.

Look for right-precordial Brugada-type visual evidence in V1:
  - J-point or ST elevation immediately after the QRS
  - a coved or convex descending ST segment, especially when followed by a
    predominantly negative T wave
  - a saddleback ST shape can be supportive but is less specific by itself
  - terminal right-precordial QRS features such as rSr' or RSR' can support the
    visual pattern, but should not be the only reason for a positive label

Return false when the beat is visually closer to a normal V1 morphology, when ST
elevation is absent, or when the image does not show enough Brugada-type
evidence. Use the support examples, when present, to calibrate the study label.

Return only valid JSON with exactly this key:

```json
{
  "clinical_brugada": true
}
```
