# External dataset strategy for IntentGlide (Person 3)

Do not blindly concatenate public datasets into the optimal-alpha regression table. They have different sensors, sampling schemes, tasks and labels. The alpha-regression target requires an observed cursor path plus a time-aligned intended/ground-truth path (or another defensible ground-truth definition).

## Roles

| Source | Available signal/labels | Appropriate use | Not appropriate for |
|---|---|---|---|
| Person 1 synthetic traces | Observed X/Y, timestamp, `truth_x/y`, trial metadata, injected disturbance settings | **Primary supervised alpha training** and controlled test cases | Claiming clinical validity |
| Balabit Mouse Dynamics | RDP mouse record/client timestamps, button/state, X/Y; challenge labels concern account/session legitimacy | External natural mouse trajectory / feature-distribution checks; test parser and robustness to real cursor dynamics | Alpha regression labels; tremor/no-tremor labels; use account identity labels as tremor labels |
| Attentive Cursor | Browser event cursor X/Y/timestamps, event/DOM fields, attention labels | Exploratory analysis of natural cursor behaviour and event handling | Resolving 4–12 Hz motion: mousemove polling is every 150 ms (about 6.67 Hz sampling; Nyquist about 3.33 Hz) |
| White & Horvitz 2019 Nature/npj study | Paper describes cursor-based tremor analysis and a 1,000-person target-selection validation study | Methodology/reference; request access to target-selection data for independent evaluation if permission is granted | Treating the paper as a freely downloadable labelled alpha dataset; data availability is restricted through Microsoft/authors |
| FittsTouch / Strathclyde 2026 | Android touchscreen task data with touch events and aligned accelerometer/gyroscope data; induced tremor rig | External modality/signal validation, after checking schema and sample timing | Directly mixing phone sensor values with PC-mouse feature rows without a modality-aware design and matched alpha labels |

## Why not merge raw datasets?

- Balabit provides no intended cursor path for each sample and its public task labels concern simulated account misuse, not motor disturbance.
- Attentive Cursor's mousemove polling interval is too slow for IntentGlide's 4–12 Hz engineering band.
- The Nature paper's underlying data is not openly downloadable; data access requires permission.
- FittsTouch is a touchscreen/IMU modality. It is useful for external checks but is not identical to mouse pointer motion.

## Recommended training/evaluation protocol

1. Train the first alpha model only on synthetic rows with Person 1's intended-path ground truth and derived optimal-alpha labels.
2. Split by complete `trial_id` / generation seed; never randomly split overlapping windows from the same reach across train and test.
3. Use Balabit and Attentive Cursor as exploratory external movement datasets for feature extraction and robustness checks only. Do not assign them synthetic `truth_x/y` or alpha labels.
4. If Nature target-selection data is obtained with permission, inspect its sampling and target geometry before deciding whether any intended-path proxy is defensible.
5. Inspect the FittsTouch data files before designing any modality transfer experiment. Do not claim it validates PC-mouse performance.
6. Report synthetic, public external, and any real-user evaluations separately.

## Sources

- Balabit repository and field definitions: https://github.com/balabit/Mouse-Dynamics-Challenge
- Attentive Cursor paper and data description: https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2020.565664/full
- Nature/npj tremor study: https://www.nature.com/articles/s41746-019-0171-4
- FittsTouch dataset: https://pureportal.strath.ac.uk/en/datasets/data-for-modelling-and-correcting-the-effects-of-tremor-on-touchs/
