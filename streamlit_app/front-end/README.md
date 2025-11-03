# FPTU Chatbot - Frontend

Modern React-based web interface for FPT University RAG Chatbot system.

## 🎨 Features

- **Interactive Chat Interface**: Real-time conversational UI with message history
- **Modern Design**: Built with React, TailwindCSS, and DaisyUI components
- **Responsive Layout**: Mobile-friendly design that works across all devices
- **Multiple Pages**: 
  - Home page with chatbot interface
  - FAQ page for common questions
  - Issue reporting page
- **Rich Text Support**: Markdown rendering for formatted responses
- **Type Animations**: Smooth typing effects for better UX
- **Navigation**: Clean navbar with routing via React Router

## 🛠️ Tech Stack

- **Framework**: React 18 with Vite
- **Styling**: TailwindCSS + DaisyUI
- **Routing**: React Router DOM
- **UI Components**: 
  - FontAwesome icons
  - React Spinners for loading states
  - React Markdown for message formatting
- **Build Tool**: Vite for fast development and optimized builds

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
npm install

# Or use the automated script
./install_and_run.sh
```

### Development

```bash
# Start development server
npm run dev

# The app will be available at http://localhost:5173
```

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## 📁 Project Structure

```
front-end/
├── src/
│   ├── components/
│   │   ├── ChatBot.jsx      # Main chatbot component
│   │   └── NavBar.jsx        # Navigation bar
│   ├── pages/
│   │   ├── HomePage.jsx      # Chat interface page
│   │   ├── FAQPage.jsx       # FAQ page
│   │   └── IssuePage.jsx     # Issue reporting page
│   ├── assets/               # Images and icons
│   ├── App.jsx               # Main app component
│   ├── App.css               # Global styles
│   └── main.jsx              # Entry point
├── public/                   # Static assets
├── index.html                # HTML template
├── vite.config.js           # Vite configuration
├── tailwind.config.cjs      # TailwindCSS config
└── package.json             # Dependencies
```

## 🔗 Backend Integration

This frontend connects to the FastAPI backend server. Make sure the backend is running before starting the frontend:

```bash
# Backend should be running at http://localhost:8000
# See main README.md for backend setup instructions
```

## 📝 Configuration

The API endpoint can be configured in the chat component to point to your backend server.

## 🎯 Usage

1. Start the backend server (see main README.md)
2. Start the frontend development server
3. Navigate to `http://localhost:5173`
4. Start chatting with the FPT University chatbot!

## 🤝 Contributing

This is part of the FPTU Chatbot project. Please refer to the main repository README for contribution guidelines.
