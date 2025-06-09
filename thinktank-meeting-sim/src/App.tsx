import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "@/components/Layout";
import Dashboard from "@/pages/Dashboard";
import Projects from "@/pages/Projects";
import Experts from "@/pages/Experts";
import Meetings from "@/pages/Meetings";
import NewProjectForm from "@/components/NewProjectForm";
import NewExpertForm from "@/components/NewExpertForm";
import NewMeetingForm from "@/components/NewMeetingForm";
import NotFound from "./pages/NotFound";
import ProjectMeetings from "@/components/ProjectMeetings";
import MeetingStream from "./components/MeetingStream";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/projects" element={<Projects />} />
            <Route path="/projects/new" element={<NewProjectForm />} />
            <Route path="/projects/:projectId/meetings" element={<ProjectMeetings />} />
            <Route path="/experts" element={<Experts />} />
            <Route path="/experts/new" element={<NewExpertForm />} />
            <Route path="/meetings" element={<Meetings />} />
            <Route path="/meetings/new" element={<NewMeetingForm />} />
            <Route path="/meeting-stream" element={<MeetingStream />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
