
import React from 'react';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Calendar, Users, Eye, Plus } from 'lucide-react';
import { Project } from '@/types';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { Meeting } from '@/types';

interface ProjectCardProps {
  project: Project;
}

const ProjectCard: React.FC<ProjectCardProps> = ({ project }) => {
  const projectMeetings = project.meetings ?? [];

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
