
import React from 'react';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Plus, Eye } from 'lucide-react';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { Meeting, Project } from '@/types';
import { getProjects } from '../../api';

const Meetings = () => {
  const [projects, setProjects] = useState<Record<string, Project>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const rawProjects = await getProjects();
        setProjects(rawProjects as Record<string, Project>);
      } catch (err) {
        console.error("Failed to fetch projects", err);
      } finally {
        setLoading(false);
      }
    };

    fetchProjects();
  }, []);

  const raw_meetings = projects ? Object.values(projects).flatMap(project => project.meetings ?? []) : [];
  const meetings = raw_meetings as Meeting[];
  

  if (loading) return <p>Loading...</p>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">All Meetings</h1>
          <p className="text-muted-foreground">Overview of all meetings across projects</p>
        </div>
        <Link to="/meetings/new">
          <Button className="flex items-center gap-2 bg-primary text-primary-foreground hover:bg-primary/90">
            <Plus className="h-4 w-4" />
            New Meeting
          </Button>
        </Link>
      </div>

      {meetings.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground mb-4">No meetings yet</p>
          <Link to="/meetings/new">
            <Button className="bg-primary text-primary-foreground hover:bg-primary/90">Create your first meeting</Button>
          </Link>
        </div>
      ) : (
        <div className="space-y-4">
          {meetings.map((meeting) => (
            <div key={meeting.id} className="border border-border rounded-lg p-4 flex items-center justify-between bg-card">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">
                  Project: {meeting.project_name} • {meeting.rounds} rounds
                </p>
                <p className="text-xs text-muted-foreground">
                  Created: {new Date(Number(meeting.timestamp)).toLocaleDateString()}
                </p>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="flex items-center gap-2 border-border text-foreground hover:bg-accent">
                  <Eye className="h-4 w-4" />
                  View
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Meetings;
