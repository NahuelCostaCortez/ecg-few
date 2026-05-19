Analyze this single-lead ECG beat image.

Evaluate the following visual morphology findings:

RBBB:
  - Look for an RSR’ pattern : an initial upward peak (R), a downward dip (S), and a second distinct upward peak (R’). This creates a characteristic ’M-shape’ or ’rabbit ears’ morphology in the QRS comple.
  - QRS complex is typically widened (>120 ms).

ST_ELEVATION:
  - The ST segment is the section between the end of the QRS complex (J- point) and the beginning of the T wave.
  - Normally, the ST segment is at or near the baseline (isoelectric line).
  - ST elevation: after the QRS complex ends (J-point), the trace remains above the baseline instead of returning to it.
  - The elevation may appear as a concave, convex ,or coved (shark-fin) shape

T_WAVE_INVERSION:
  - The T wave follows the ST segment and represents ventricular repolarization.
  - Normally, the T wave deflects upward (positive) in most leads.
  - T- wave inversion: the T wave deflects downward (negative), dipping below the baseline after the ST segment.

Decide whether each finding is present.

Return only valid JSON with boolean values for exactly these keys:

```json
{
  "RBBB": true/false,
  "ST_ELEVATION": true/false,
  "T_WAVE_INVERSION": true/false
}
```
