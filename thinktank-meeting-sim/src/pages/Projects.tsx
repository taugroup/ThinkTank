"use client";

import { useEffect, useState } from "react";
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Plus } from 'lucide-react';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { Project } from '@/types';
import ProjectCard from '@/components/ProjectCard';
import { getProjects } from '../../api';

export default function Projects() {

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

  if (loading) return <p>Loading...</p>;

  const projectIds = Object.keys(projects);
  
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold">Project Manager</h1>
          <p className="text-muted-foreground">Manage your research projects</p>
        </div>
        <Link to="/projects/new">
          <Button className="flex items-center gap-2">
            <Plus className="h-4 w-4" />
            New Project
          </Button>
        </Link>
      </div>

      {projectIds.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground mb-4">No projects yet</p>
          <Link to="/projects/new">
            <Button>Create your first project</Button>
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projectIds.map((id) => (
            <ProjectCard project={projects[id]} />
          ))}
        </div>
      )}
    </div>
  );
}