Analyze this single-lead ECG beat image.

Evaluate the following visual morphology findings:

- RBBB: look for a widened terminal QRS morphology with an R-prime-like positive terminal deflection.
- ST_ELEVATION: look for elevation of the J-point or ST segment above the local baseline after the QRS complex.
- T_WAVE_INVERSION: look for a predominantly negative T wave after the ST segment.

Decide whether each finding is present.

Return only valid JSON with boolean values for exactly these keys:

```json
{
  "RBBB": true,
  "ST_ELEVATION": false,
  "T_WAVE_INVERSION": true
}
```
