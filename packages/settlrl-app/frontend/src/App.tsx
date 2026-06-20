import { Routes, Route, Navigate } from "react-router-dom";
import Menu from "./pages/Menu";
import PlayView from "./pages/PlayView";
import LobbyView from "./pages/LobbyView";
import LeaderboardView from "./pages/LeaderboardView";
import ProfileView from "./pages/ProfileView";
import ReplayView from "./pages/ReplayView";
import HelpView from "./pages/HelpView";
import AuthView from "./pages/AuthView";
import AdminView from "./pages/AdminView";
import AdminLink from "./components/AdminLink";

export default function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<Menu />} />
        <Route path="/play" element={<PlayView />} />
        <Route path="/play/:id" element={<PlayView />} />
        <Route path="/lobby" element={<LobbyView />} />
        <Route path="/help" element={<HelpView />} />
        <Route path="/leaderboard" element={<LeaderboardView />} />
        <Route path="/profile" element={<ProfileView />} />
        <Route path="/replay" element={<ReplayView />} />
        <Route path="/admin" element={<AdminView />} />
        <Route path="/login" element={<AuthView initialMode="login" />} />
        <Route path="/register" element={<AuthView initialMode="register" />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <AdminLink />
    </>
  );
}
