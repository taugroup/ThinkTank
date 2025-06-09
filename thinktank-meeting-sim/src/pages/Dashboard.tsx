
import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { FolderOpen, Users, Calendar, Plus } from 'lucide-react';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { Project } from '@/types';
import { getProjects, getExpertTemplates } from '../../api'

type StatItem = {
  title: string
  value: number
  icon: React.ComponentType<{ className?: string }>
  href: string
}

const Dashboard = () => {
  const [stats, setStats] = useState<StatItem[]>([
    { title: 'Total Projects', value: 0, icon: FolderOpen, href: '/projects' },
    { title: 'Total Experts',  value: 0, icon: Users,      href: '/experts' },
    { title: 'Total Meetings', value: 0, icon: Calendar,   href: '/meetings' },
  ])
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading]   = useState<boolean>(true)
  const [error, setError]       = useState<string | null>(null)

  useEffect(() => {
    const fetchAll = async () => {
      try {
        setLoading(true)
        // fire all three in parallel
        const [rawProjs, experts] = await Promise.all([
          getProjects(),
          getExpertTemplates(),

        ])
        const projs = rawProjs as Project; 
        const projList = Object.values(projs);
        const meetings = projList.reduce(
          (sum, proj) => sum + (proj.meetings?.length ?? 0),
          0
        );
        setProjects(projList)
        setStats([
          {
            title: 'Total Projects',
            value: projList.length,
            icon: FolderOpen,
            href: '/projects',
          },
          {
            title: 'Total Experts',
            value: experts.length,
            icon: Users,
            href: '/experts',
          },
          {
            title: 'Total Meetings',
            value: meetings,
            icon: Calendar,
            href: '/meetings',
          },
        ])
        setError(null)
    } catch (err: any) {
      console.error(err)
      setError('Failed to load stats.')
    } finally {
      setLoading(false)
    }
  }

  fetchAll()
}, [])

if (loading) return <div>Loading stats…</div>
if (error)   return <div style={{ color: 'red' }}>{error}</div>

return (
  <div className="space-y-8">
    {/* Header */}
    <div>
      <h1 className="text-2xl font-semibold">Dashboard</h1>
      <p className="text-muted-foreground">
        Multi-agent research simulation platform
      </p>
    </div>

    {/* Stats Grid */}
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {stats.map((stat) => {
        const Icon = stat.icon
        return (
          <Link key={stat.title} to={stat.href}>
            <Card className="hover:bg-accent/50 transition-colors cursor-pointer">
              <CardHeader className="flex items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium">
                  {stat.title}
                </CardTitle>
                <Icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stat.value}</div>
              </CardContent>
            </Card>
          </Link>
        )
      })}
    </div>
    {/* Quick Actions */}
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
    </div>
  )
}

export default Dashboard;
