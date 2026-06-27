Analyze this single-beat right precordial ECG image from lead {lead}.

Evaluate these morphology findings independently:

RBBB:
  - terminal RSR' or rSr' morphology in the QRS complex
  - delayed terminal right-precordial activation compatible with right bundle branch block

ST_ELEVATION:
  - J-point or ST segment remains elevated above the baseline after the QRS complex
  - coved, convex, or saddleback elevation can count when visually present

T_WAVE_INVERSION:
  - the T wave after the ST segment is predominantly negative below baseline

Return only valid JSON with boolean values for exactly these keys:

```json
{
  "RBBB": true,
  "ST_ELEVATION": true,
  "T_WAVE_INVERSION": true
}
```
