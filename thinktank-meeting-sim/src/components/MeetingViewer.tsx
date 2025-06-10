import React, { useEffect, useState, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Project, Expert } from '@/types';
import ReactMarkdown from 'react-markdown';
import { getProjectByName } from '../../api';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';


const MeetingViewer= () => {
    const { projectTitle } = useParams<{ projectTitle: string }>();
    const {meetingTopic } = useParams<{ meetingTopic: string }>();

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

    const meeting = project?.meetings?.find(m => m.meeting_topic === meetingTopic);

    if (!meeting) {
        return <p>Meeting not found.</p>;
    }

    return (
        <div className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8 space-y-8">
            <Link to={`/projects/${project.title}/meetings`}>
                <Button variant="outline" size="sm" className="border-border text-foreground hover:bg-accent">
                    <ArrowLeft className="h-4 w-4" />
                </Button>
            </Link>
            {/* ====== 1. Project Header ====== */}
            <div className="border-b pb-4">
                <h1 className="text-3xl font-bold tracking-tight text-white-900 sm:text-4xl">
                    {project.title}
                </h1>
                <p className="mt-2 text-lg text-white-600">
                    {project.description}
                </p>
            </div>

            {/* ====== 2. Experts Table ====== */}
            <div className="space-y-4">
                <h2 className="text-2xl font-semibold text-white-800">Experts Involved</h2>
                <div className="overflow-x-auto rounded-lg border">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                            <tr>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Title</th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Role</th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Expertise</th>
                                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Goal</th>
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            {meeting.experts.map((expert: Expert, idx: number) => (
                                <tr key={idx}>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{expert.title}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{expert.role}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{expert.expertise}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{expert.goal}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
            {/* ====== 3. Meeting Transcript ====== */}
            <div className="space-y-4">
                <h2 className="text-2xl font-semibold text-white-800">
                    Transcript: <span className="font-normal"></span>
                </h2>
                <h2 className="text-lg text-white-400">
                    {meeting.meeting_topic}
                </h2>
                <div className="space-y-6 bg-gray-800 text-white p-6 rounded-lg max-h-[75vh] overflow-y-auto shadow-inner">
                    {meeting.transcript.map((msg: { name: string; content: string }, idx: number) => (
                        <div key={idx} className="border-b border-gray-600 pb-4 last:border-b-0">
                            <ReactMarkdown className="prose prose-sm prose-invert max-w-none">
                                {`## ${msg.name}\n\n${msg.content}`}
                            </ReactMarkdown>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

export default MeetingViewer