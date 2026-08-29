# Getting creditlib onto GitHub

Follow in order. The step that trips everyone is #6 (auth) — read it before you start.

---

## 1. GitHub account

github.com → sign up. Use an email you'll keep.

**Then verify the email** (check your inbox, click the link). Unverified emails cause
step 7 to fail silently — your commits push but don't appear on your profile.

## 2. Install Git

**Mac** — open Terminal:
```bash
git --version
```
If it prompts to install Command Line Tools, accept. That's Git.

**Windows** — download from git-scm.com, install with defaults, then use **Git Bash**
(not Command Prompt) for everything below.

## 3. Tell Git who you are

Use the **same email as your GitHub account**, or your commits won't be attributed to
your profile — which defeats the purpose of a portfolio repo.

```bash
git config --global user.name "Your Full Name"
git config --global user.email "the-email-on-your-github-account@example.com"
```

## 4. Create the empty repo on GitHub

github.com → **+** (top right) → **New repository**

- Name: `creditlib`
- Description: `Single-name CDS pricing engine: hazard-rate bootstrapping, exact leg integration, credit risk measures`
- **Public**
- **Do NOT tick** "Add a README", "Add .gitignore", or "Choose a license"

That last point matters. If GitHub creates files, your first push is rejected as a
conflict and you'll be untangling it instead of shipping.

Leave the page open — you'll need the URL.

## 5. Initialise locally

Unzip `creditlib.zip` somewhere sensible, then:

```bash
cd path/to/creditlib          # the folder containing README.md and pyproject.toml
git init
git add -A
git status                    # LOOK AT THIS BEFORE COMMITTING
```

`git status` should list roughly 25 files. If you see `.venv/`, `__pycache__/`,
`creditlib.egg-info/` or `.pytest_cache/`, stop — `.gitignore` isn't being applied:

```bash
git rm -r --cached . && git add -A && git status
```

Then:
```bash
git commit -m "creditlib: single-name CDS pricing engine with derivations and validation"
```

## 6. Authentication — read this first

**GitHub does not accept your password.** If you're prompted for one, you need a
Personal Access Token instead.

github.com → click your avatar → **Settings** → scroll to **Developer settings**
(very bottom of the left sidebar) → **Personal access tokens** → **Tokens (classic)**
→ **Generate new token (classic)**

- Note: `creditlib push`
- Expiration: 90 days
- Scope: tick **`repo`** (just that one)
- Generate, then **copy the token immediately** — it is shown once and never again.

Paste it into a password manager or a note. When Git asks for a *password*, paste the
token. Your GitHub username is still your username.

## 7. Push

```bash
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/creditlib.git
git push -u origin main
```

Username → your GitHub username. Password → **the token from step 6**.

To avoid re-entering it every time:
```bash
git config --global credential.helper store      # Linux/Windows
git config --global credential.helper osxkeychain # Mac
```

## 8. Check it worked

Open `github.com/YOUR-USERNAME/creditlib`. Confirm:

- [ ] README renders with the test-count block at the top
- [ ] `.venv/` and `__pycache__/` are **absent**
- [ ] `docs/creditlib_documentation.pdf` opens in the browser
- [ ] Your avatar appears next to the commit (if not, step 3's email is wrong)

## 9. Fix the Colab placeholders

`notebooks/creditlib_colab.ipynb` contains `YOUR-USERNAME` in four places. Replace them:

```bash
# Mac
sed -i '' 's/YOUR-USERNAME/your-actual-username/g' notebooks/creditlib_colab.ipynb README.md
# Linux / Git Bash on Windows
sed -i 's/YOUR-USERNAME/your-actual-username/g' notebooks/creditlib_colab.ipynb README.md

git add -A && git commit -m "Point Colab notebook at the repo" && git push
```

## 10. Get your shareable link

```
https://colab.research.google.com/github/YOUR-USERNAME/creditlib/blob/main/notebooks/creditlib_colab.ipynb
```

**Test it in a private/incognito window.** If it works logged out, it works for anyone.
That link is the thing you put in a LinkedIn comment or an email to a contact.

## 11. Two minutes of polish

On the repo page, click the gear next to **About** (right sidebar):

- Description: as in step 4
- Website: paste the Colab link
- Topics: `credit-derivatives` `fixed-income` `quantitative-finance` `cds`
  `structured-credit` `python` `quantlib`

Topics are how people browsing GitHub find you. Free, thirty seconds.

## 12. Later — making changes

```bash
git add -A
git commit -m "what changed and why"
git push
```

Write commit messages a stranger could follow. They're public, and on a portfolio repo
they're read.

---

## If something breaks

| Message | Fix |
|---|---|
| `remote: Support for password authentication was removed` | You used a password. Go to step 6 and use a token. |
| `Updates were rejected because the remote contains work` | You let GitHub create a README. Easiest fix: delete the repo on GitHub, redo step 4 with nothing ticked. |
| `fatal: remote origin already exists` | `git remote set-url origin https://github.com/YOUR-USERNAME/creditlib.git` |
| `src refspec main does not match any` | You haven't committed yet. Run step 5. |
| Push is enormous / very slow | `.venv` got committed. `git rm -r --cached .venv && git commit -m "drop venv" && git push` |
| Commits don't show on your profile | Email in step 3 doesn't match a **verified** email on your GitHub account. |
