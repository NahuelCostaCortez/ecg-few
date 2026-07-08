Analyze this single-beat right precordial ECG image from lead {lead}.

Inspect the beat from left to right:
  - identify the QRS complex, including any second positive terminal deflection
  - use the flat PR/TP segment around the beat as the visual baseline when visible
  - inspect the J point and ST segment immediately after the QRS
  - inspect the T wave that follows the ST segment

Evaluate these morphology findings independently:

RBBB:
  - true when the QRS has a right-precordial terminal conduction pattern such as
    RSR', rSr', rsR', or a clear late positive R' after the initial QRS deflection
  - also true when the terminal part of the QRS looks delayed or widened in a way
    compatible with right bundle branch block morphology
  - false when there is no clear terminal R' or delayed terminal QRS morphology

ST_ELEVATION:
  - true when the J point or ST segment after the QRS remains above the visual
    baseline
  - coved/convex elevation and saddleback elevation count when visually present
  - false when the ST segment returns to baseline or is not visibly elevated

T_WAVE_INVERSION:
  - true when the T wave after the ST segment is predominantly negative, below
    the visual baseline
  - false when the T wave is upright, flat without a clear negative component, or
    not visible enough to judge as inverted

Return only valid JSON with boolean values for exactly these keys:

```json
{
  "RBBB": true,
  "ST_ELEVATION": true,
  "T_WAVE_INVERSION": true
}
```
