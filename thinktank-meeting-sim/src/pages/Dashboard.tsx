
import React from 'react';
import { Link } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { FolderOpen, Users, Calendar, Plus } from 'lucide-react';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { Project } from '@/types';

const Dashboard = () => {
  const [projects] = useLocalStorage<Project[]>('projects', []);

  const stats = [
    {
      title: 'Total Projects',
      value: projects.length,
      icon: FolderOpen,
      href: '/projects',
    },
    {
      title: 'Total Experts',
      value: projects.reduce((acc, project) => acc + project.experts.length, 0),
      icon: Users,
      href: '/experts',
    },
    {
      title: 'Total Meetings',
      value: projects.reduce((acc, project) => acc + project.meetings.length, 0),
      icon: Calendar,
      href: '/meetings',
    },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <p className="text-muted-foreground">Multiagent research simulation platform</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <Link key={stat.title} to={stat.href}>
              <Card className="hover:bg-accent/50 transition-colors cursor-pointer">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
                  <Icon className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{stat.value}</div>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>

      <div className="space-y-4">
        <h2 className="text-lg font-medium">Quick Actions</h2>
        <div className="flex gap-4">
          <Link to="/projects/new">
            <Button className="flex items-center gap-2">
              <Plus className="h-4 w-4" />
              New Project
            </Button>
          </Link>
          <Link to="/experts/new">
            <Button variant="outline" className="flex items-center gap-2">
              <Plus className="h-4 w-4" />
              New Expert
            </Button>
          </Link>
        </div>
      </div>

      {projects.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-medium">Recent Projects</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {projects.slice(0, 3).map((project) => (
              <Link key={project.id} to={`/projects/${project.id}`}>
                <Card className="hover:bg-accent/50 transition-colors cursor-pointer">
                  <CardHeader>
                    <CardTitle className="text-base">{project.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {project.description}
                    </p>
                    <div className="flex justify-between items-center mt-4 text-xs text-muted-foreground">
                      <span>{project.experts.length} experts</span>
                      <span>{project.meetings.length} meetings</span>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
