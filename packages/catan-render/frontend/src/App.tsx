import { Routes, Route, Navigate } from "react-router-dom";
import Menu from "./pages/Menu";
import PlayView from "./pages/PlayView";
import ReplayView from "./pages/ReplayView";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Menu />} />
      <Route path="/play" element={<PlayView />} />
      <Route path="/replay" element={<ReplayView />} />
      <Route path="/replay/:gameId" element={<ReplayView />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
