# GlassBox-BI — Frontend Dashboard

This directory contains the Next.js TypeScript web application for the **GlassBox-BI** platform.

## Overview
- **Framework**: Next.js (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom glassmorphism tokens
- **Backend API URL**: Configured via `NEXT_PUBLIC_API_URL` (default: `http://127.0.0.1:8000`)

## Getting Started

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env.local`:
```bash
cp .env.example .env.local
```

### 3. Run Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 4. Build for Production
```bash
npm run build
npm start
```
