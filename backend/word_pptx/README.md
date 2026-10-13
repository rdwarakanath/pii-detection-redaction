# SecureDoc Quick Start

Detect and redact supported PII in the supplied PPTX and DOCX samples using the placeholder `xxxxxxxxxx`. Run these commands in Windows PowerShell from the project root. Use Python 3.13 for the pinned environment.

## 1. Set up the project

For a fresh checkout:

```powershell
git clone --branch Swapnil https://github.com/rdwarakanath/pii-detection-redaction.git
cd pii-detection-redaction
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

For the existing local project, start with:

```powershell
cd D:\securedoc
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow scripts for this terminal session only, then retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 2. Add the input files

Original samples are excluded from Git. Create the folder and copy the provided files into it:

```powershell
New-Item -ItemType Directory -Force data/input | Out-Null
```

- `data/input/Cadence_Financial_Group_Organizational_Pack.pptx`
- `data/input/Cadence_TPRM_Training.docx`

Keep `config/docx_image_redactions.json` in place for the reviewed DOCX screenshot masks. Close the output documents in Word and PowerPoint before regenerating them.

## 3. Check extraction and detection

```powershell
python -m tests.test_extractors
python -m tests.test_presidio_patterns
```

## 4. Redact and audit the samples

```powershell
# PowerPoint: generate the redacted copy, then audit its text
python -m tests.test_pptx_redactor
python -m tests.audit_redacted_pptx

# Word: generate the redacted copy, check image masks, then audit
python -m tests.test_docx_redactor
python -m tests.test_docx_image_redaction
python -m tests.audit_redacted_docx
```

Run the commands in order and investigate any failure before using the output. These entry points are tests and demonstrations for the supplied samples, not a general upload interface.

## 5. Find the results

- `data/output/Cadence_Financial_Group_Organizational_Pack_redacted.pptx`
- `data/output/Cadence_TPRM_Training_redacted.docx`
- `data/output/Cadence_TPRM_Training_redacted.review.json` (local review report containing original identifiers; excluded from Git)

Inspect regenerated documents visually. DOCX screenshot masks apply only to images matching the reviewed fingerprints; new or changed images need review. Passing tests or finding no regex matches does not guarantee complete PII removal. An unresolved review must be completed before sending document content to an LLM.
