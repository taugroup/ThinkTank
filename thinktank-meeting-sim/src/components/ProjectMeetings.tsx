
import React from 'react';
import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Plus, Eye, ArrowLeft } from 'lucide-react';
import { getProjectByName } from '../../api';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { Meeting, Project } from '@/types';

const ProjectMeetings = () => {
  const { projectTitle } = useParams<{ projectTitle: string }>();

  // 2. Set up state to hold the full project object, which starts as `null`.
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Make sure we have the title before trying to fetch.
    if (!projectTitle) {
      setError("No project specified.");
      setLoading(false);
      return;
    }
    const fetchProjectData = async () => {
      try {
        setLoading(true);
        // 3. Use the identifier to fetch the full project object.
        const fetchedProject = await getProjectByName(projectTitle);
        setProject(fetchedProject);
      } catch (err) {
        console.error("Failed to fetch project:", err);
        setError("Could not load the project.");
      } finally {
        setLoading(false);
      }
    };

    fetchProjectData();
  }, [projectTitle]);

  if (loading) {
    return <p>Loading project...</p>;
  }

  if (error) {
    return <p className="text-red-500">{error}</p>;
  }

  // If loading is done and there's still no project, it wasn't found.
  if (!project) {
    return <p>Project not found.</p>;
  }

  const projectMeetings = project.meetings || [];
  console.log("Project meetings:", projectMeetings);

  if (!project) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">Project not found</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link to="/projects">
          <Button variant="outline" size="sm" className="border-border text-foreground hover:bg-accent">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-semibold text-foreground">{project.title} - Meetings</h1>
          <p className="text-muted-foreground">View and manage meetings for this project</p>
        </div>
        <Link to="/meetings/new">
          <Button className="flex items-center gap-2 bg-primary text-primary-foreground hover:bg-primary/90">
            <Plus className="h-4 w-4" />
            New Meeting
          </Button>
        </Link>
      </div>

      {projectMeetings.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground mb-4">No meetings for this project yet</p>
          <Link to="/meetings/new">
            <Button className="bg-primary text-primary-foreground hover:bg-primary/90">Create your first meeting</Button>
          </Link>
        </div>
      ) : (
        <div className="space-y-4">
          {projectMeetings.map((meeting) => (
            <div key={meeting.id} className="border border-border rounded-lg p-4 flex items-center justify-between bg-card">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">
                  • {meeting.meeting_topic}
                </p>
                <p className="text-sm text-muted-foreground">
                  {meeting.rounds} rounds
                </p>
                <p className="text-xs text-muted-foreground">
                  Created: {new Date(Number(meeting.timestamp)).toLocaleDateString()}
                </p>
              </div>
              <div className="flex gap-2">
                <Link to={`/meeting/${projectTitle}/${meeting.meeting_topic}`}>
                  <Button variant="outline" size="sm" className="flex items-center gap-2 border-border text-foreground hover:bg-accent">
                    <Eye className="h-4 w-4" />
                    View
                  </Button>
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ProjectMeetings;
