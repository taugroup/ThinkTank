
import React from 'react';
import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Calendar, Users, Eye, Plus } from 'lucide-react';
import { Project } from '@/types';
import { getProjectByName } from '../../api';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { Meeting } from '@/types';

interface ProjectCardProps {
  project: Project;
}

const ProjectCard: React.FC<ProjectCardProps> = ({ project }) => {
  const project_title = project.title;
  const [project_new, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Ensure project_title is available before fetching
    if (!project_title) {
      setLoading(false);
      setError("No project title provided in URL.");
      return;
    }

    const fetchProject = async () => {
      try {
        setLoading(true);
        const data = await getProjectByName(project_title);
        // 3. Set the project state with the returned data
        setProject(data); 
      } catch (err) {
        console.error("Failed to fetch project", err);
        setError("Could not load project details. Please try again later.");
      } finally {
        setLoading(false);
      }
    };
    fetchProject();
  }, [project_title]);

  if (loading) {
    return <p className="text-center mt-8">Loading project details...</p>;
  }

  // Handle error state
  if (error) {
    return <p className="text-center mt-8 text-red-500">{error}</p>;
  }

  // Handle case where project was not found after loading
  if (!project) {
    return <p className="text-center mt-8">Project not found.</p>;
  }
  const projectMeetings = project_new.meetings || [];

  return (
    <Card className="hover:bg-accent/50 transition-colors bg-card border-border">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="text-base font-medium text-foreground">{project.title}</CardTitle>
            <p className="text-sm text-muted-foreground line-clamp-2">
              {project.description}
            </p>
          </div>
          <Badge variant="secondary" className="text-xs bg-secondary text-secondary-foreground">
            {projectMeetings.length}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="pt-0 space-y-4">
        
        <div className="flex gap-2">
          <Link to={`/projects/${project.title}/meetings`} className="flex-1">
            <Button variant="outline" size="sm" className="w-full flex items-center gap-2 border-border text-foreground hover:bg-accent">
              <Eye className="h-3 w-3" />
              View Meetings
            </Button>
          </Link>
          <Link to={`/meetings/new`}>
            <Button size="sm" className="flex items-center gap-2 bg-primary text-primary-foreground hover:bg-primary/90">
              <Plus className="h-3 w-3" />
              Meeting
            </Button>
          </Link>
        </div>
      </CardContent>
    </Card>
  );
};

export default ProjectCard;
