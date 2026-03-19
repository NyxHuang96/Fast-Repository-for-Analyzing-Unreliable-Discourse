---
title: "tutorial.md"
author: "Marco Wang"
date: "2026-03-18"
Disclaimer: This documentation is generated with the help of Gemini3
---

# How to Launch the FRAUD Application via Docker

You can easily launch the entire FRAUD application (both the backend API and the frontend) entirely inside a Docker container.

## Step 1: Load the Docker Image

First, load the image into your local Docker instance. Download the `fraud.tar` file from the [Google Drive link](https://drive.google.com/file/d/1HBUaI_rFJyFeoUFhfaiTQuKFJjZoC744/view?usp=sharing) and save it to your computer. Then, open your terminal in the folder containing `fraud.tar` and run:

```bash
docker load -i fraud.tar
```

*What this does:* This command tells Docker to read the `.tar` file and extract the image layers and tags. Once it finishes, the image named `fraud` will be available in your local Docker image registry.

## Step 2: Run the Docker Container

Once the image is loaded, spin up the application container:

```bash
docker run -p 8000:8000 fraud
```

*What this does:* 
- `docker run` tells Docker to start a new container from the `fraud` image. 
- `-p 8000:8000` tells Docker to map port 8000 on your local machine to port 8000 inside the Docker container (which is where the FastAPI server is running).

## Step 3: Access the Application

You can now open your web browser and navigate directly to: [**http://127.0.0.1:8000**](http://127.0.0.1:8000){.uri}

---

# 🧪 Peer Review Testing Guide

We kindly ask our peer reviewers to explore and evaluate the following implemented features. Please cross-reference the front-end interface with the back-end endpoints to verify smooth data integration!

## 🖥️ Front-End Features to Test

### 1. Interactive Search & Data Badges
- **Action:** Navigate to the **Search** page and type a query (e.g., "bank", "account", or "research") and click Search.
- **Expected Result:** Results should appear immediately. Notice that every result card includes visual data badges highlighting whether the document is `Raw Text` vs `✓ Annotated`, along with its label (e.g., `SPAM`, `HAM`). The results show bilingual search snippets (EN and ZH).

### 2. Full-Text Context Modals
- **Action:** Click anywhere inside one of the individual Search Result cards.
- **Expected Result:** A dark-mode pop-up modal overlay should instantly appear, displaying the complete textual context of the English and Chinese versions of that document. You can close the modal via the "X" button, clicking outside the window, or pressing the `Escape` key.

### 3. Recent Searches Dropdown
- **Action:** Perform 2 to 3 different search queries in the main search bar. Then, click the **"Recent Searches ▼"** text below the search bar on the right side.
- **Expected Result:** A dropdown menu will appear showing your search history! It securely stores up to 8 recent searches. Click on an old keyword to auto-fill and jump back to that query, or click **"Clear all"** to verify that it deletes the local history perfectly.

### 4. Interactive Sidebar Filtering (Corpora & Labels)
- **Action:** Look at the left sidebar under "Corpora". Click the different corpus targets: **All**, **Annotated**, or **Raw**. Then, test clicking **Ham**, **Spam**, or **Phishing** in the Labels section below it.
- **Expected Result:** The clicked parameters will be dynamically highlighted in blue as active filters. The next time you hit "Search", your query will actively be constrained exactly to those datasets and labels without refreshing the page!

### 5. Dynamic Sidebar Statistics & Tooltips
- **Action:** Notice the numbers inside the parentheses next to the corpora links on the left sidebar (e.g., `(53,414 Docs)`).
- **Expected Result:** These numbers are not hardcoded! Upon page loading, they dynamically `fetch()` live counting metrics natively from our backend index API.
- **Action:** Hover your mouse over the Top Navigation links or the active elements to reveal their brief accessibility tooltips.

### 6. Interactive Chart.js Statistics Page
- **Action:** Navigate to the **Statistics** page using the top navigation bar.
- **Expected Result:** You should see multiple visually appealing and interactive charts. Try using the dropdowns above the charts to toggle between Doughnut, Bar, Pie, and Polar Area graphs, or test the filter buttons (English, Chinese, All).


## ⚙️ Back-End Features & API to Test

For reviewers interested in testing the API directly via cURL or Postman (or directly in your browser), the FastAPI backend provides several endpoints powering the front-end features above.

### 1. The `/stats` Endpoint
- **Action:** Visit `http://127.0.0.1:8000/stats` in your browser.
- **Expected Result:** The backend queries the complete Whoosh index and returns a JSON object containing accurate values for `total_docs`, `annotated_docs`, `raw_docs`, and an exact enumeration of all validation labels (`Ham`, `Spam`, `Phish`). This dynamically populates the Statistics page!

### 2. The `/doc` Full-Text Endpoint
- **Action:** Navigate to `http://127.0.0.1:8000/doc/1` in your browser (assuming document ID 1 exists).
- **Expected Result:** It fetches and returns the raw underlying text for both English and Chinese corresponding to that unique Document ID. This is how the Front-End Full-Text Modal operates!

### 3. The `/search` Endpoint
The `/search` endpoint natively supports URL parameters to filter Whoosh indexes securely.
- **Action:** Append query parameters directly against the backend. Try an API call such as: 
  `http://127.0.0.1:8000/search?q=bank&annotated_only=true&label=spam`
- **Expected Result:** You should receive a JSON response containing an array of results, but exclusively returning documents with `is_annotated=true` and tagged under the `spam` label. This proves the Sidebar Filtering logic is perfectly coupled to the backend logic.
