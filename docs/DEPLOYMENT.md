# Deploying to Streamlit Community Cloud

This guide publishes the dashboard from GitHub to [Streamlit Community Cloud](https://share.streamlit.io)
(free). No secrets or API keys are needed.

## 1. Prepare the repository

1. **Get the code on the branch you will deploy.** Development happened on
   `claude/gallant-hawking-2exgyr`. Either merge it into `main` with a pull request (recommended)
   or deploy that branch directly in step 3.
2. **(Recommended) Bundle the dataset** so visitors can click *Use bundled UCI dataset* instead of
   uploading. The UCI data is licensed CC BY 4.0 (redistribution with attribution is allowed, and
   the app cites it). `.gitignore` already allows exactly these two files:

   ```bash
   git add data/raw/student-mat.csv data/raw/student-por.csv
   ```
   ```bash
   git commit -m "Add UCI Student Performance dataset (CC BY 4.0)"
   ```
   ```bash
   git push
   ```
3. **Check it locally first** (from a fresh clone, in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```
   ```bash
   pytest
   ```
   ```bash
   streamlit run app.py
   ```

Files the cloud build relies on, all already in the repository root:

| File | Purpose |
|---|---|
| `app.py` | Entry point (the "main file path") |
| `requirements.txt` | Pinned dependencies, installed automatically |
| `.streamlit/config.toml` | Theme and server settings, picked up automatically |
| `app_pages/`, `src/` | Pages and application code |

## 2. Create the Streamlit Cloud account

1. Go to <https://share.streamlit.io> and **sign in with GitHub**.
2. Authorise Streamlit to read your repositories. For a private repository, grant access to it
   (or to the organisation that owns it) when prompted.

## 3. Deploy the app

1. Click **Create app** → deploy from a GitHub repository.
2. Fill in:
   - **Repository:** `vishalchaudhary013/rohti-project` (or your fork)
   - **Branch:** `main` (or `claude/gallant-hawking-2exgyr` if you did not merge)
   - **Main file path:** `app.py`
   - **App URL:** choose a subdomain, e.g. `education-analytics`
3. Open **Advanced settings** and select **Python 3.12** (any of 3.10–3.13 works with the pinned
   versions). The Python version is fixed at deploy time; to change it later you must delete and
   redeploy the app. Leave *Secrets* empty.
4. Click **Deploy**. The first build installs `requirements.txt` and takes a few minutes; follow
   it in the log panel. When it finishes, the app opens at your chosen URL.

## 4. Restrict who can see it (recommended)

Student data is personal data. In the app's **Settings → Sharing**, make the app private and invite
only the people who should use it. Reasons:

- All viewers share **one server process and one disk**. The upload history, the last processed
  dataset (`data/processed/`) and the saved Early Warning model (`models/`) are visible to every
  viewer of the app. Uploaded data itself is kept per browser session.
- The cloud disk is **ephemeral**: SQLite history, saved models and processed files are wiped when
  the app restarts, redeploys or wakes from sleep. Bundled files committed to Git always survive.

For multi-user production use, move storage to a managed database and per-user model storage (see
README → Next steps).

## 5. Smoke-test checklist

Run through this after every deploy (about 5 minutes):

- [ ] **Home** loads within a few seconds and shows "No dataset loaded" with an *Upload data* link.
- [ ] **Data Ingestion → Use bundled UCI dataset** (if bundled) or upload `student-mat.csv` and
      `student-por.csv`: the subject is detected per file, *Validate and load* shows
      **2 / 2 files accepted** and **395 + 649 = 1,044** rows for the UCI files, plus the
      data-quality flags and an upload-history entry.
- [ ] Upload a deliberately broken file (e.g. delete the `G3` column): it is **rejected** with a
      readable reason and no stack trace.
- [ ] **Home** now shows records, average G3 and pass rate.
- [ ] **Executive Overview**: KPI tiles, pass/fail and subject donuts, bar charts, tables. Change a
      sidebar filter (e.g. Subject = Mathematics); numbers update and the filter persists when you
      switch pages.
- [ ] Turn on **Exclude G3 = 0**; the record count drops by the G3 = 0 count shown on Data Ingestion.
- [ ] **Exploratory Analysis**: all four tabs render; fail bars are hatched orange; *Show data*
      opens tables; **Export all figures** offers a zip with 12 PNGs.
- [ ] **Model Evaluation**: the first load takes about 15–30 s, then the table shows 8 rows (3 models +
      baseline × 2 settings). Moving the threshold slider updates the KPIs, the confusion matrix
      and the trade-off line. **Save** shows a success message.
- [ ] **Early Warning**: uses the saved model, shows tier KPIs, the risk donut, the at-risk table
      with three reasons per student; search and tier filter work; **Download as CSV** works;
      a row appears in *Prediction history*.
- [ ] **Methodology** renders fully.
- [ ] Check the app logs (bottom-right *Manage app*) for warnings or errors.

## 6. Updating

Push to the deployed branch; Streamlit Cloud redeploys automatically. Use **Manage app → Reboot**
to restart, or after changing `requirements.txt` if the build cache gets stuck.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails installing a package | Confirm the Python version chosen in Advanced settings is 3.10–3.13; the pins in `requirements.txt` were tested on 3.10 and 3.13. |
| `ModuleNotFoundError: src` | The main file path must be `app.py` at the repository root. |
| Page not found for a sidebar entry | Paths are case-sensitive: pages live in `app_pages/` with the exact file names used in `app.py`. |
| *Use bundled UCI dataset* button missing | The two CSVs are not committed to `data/raw/` (see step 1.2). |
| Early Warning says no model is saved | Expected after a restart (ephemeral disk). Save one on Model Evaluation, or click *Train recommended model*. |
| App is slow or restarts during Model Evaluation | Community Cloud has limited memory and CPU. Use the sidebar filters to evaluate a smaller subset, or run `python scripts/train.py` locally. |
| App is asleep | Community Cloud apps sleep after a period without visitors; open the URL and click to wake it. |
