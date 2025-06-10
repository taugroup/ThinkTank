
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { Expert, Meeting, Project } from '@/types';
import { Upload, X } from 'lucide-react';
import { getProjects, getExpertTemplates } from '../../api'; 
import { get } from 'http';

const NewMeetingForm = () => {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Record<string, Project>>({});
  const [experts, setExperts] = useState<Expert[]>([]);
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(true);


  const [meetings, setMeetings] = useLocalStorage<Meeting[]>('meetings', []);
  const [selectedProjectId, setSelectedProjectId] = useState<string>('');
  const [title, setTitle] = useState('');
  const [rounds, setRounds] = useState(1);
  const [selectedExperts, setSelectedExperts] = useState<Expert[]>([]);
  const [files, setFiles] = useState<Record<string, FileList>>({});
  const [expertToAdd, setExpertToAdd] = useState<string>('');

  useEffect(() => {
    const fetchProjectsAndExperts = async () => {
      try {
        const rawProjects = await getProjects();
        setProjects(rawProjects as Record<string, Project>);
        const rawExperts = await getExpertTemplates();
        setExperts(rawExperts as Expert[]);
      } catch (err) {
        console.error("Failed to fetch projects or experts", err);
      } finally {
        setLoading(false);
      }
    };
    fetchProjectsAndExperts();
  }, []);

  const handleAddExpert = () => {
    if (expertToAdd) {
      const expert = experts.find(e => e.title === expertToAdd);
      if (expert && !selectedExperts.find(e => e.title === expert.title)) {
        setSelectedExperts([...selectedExperts, expert]);
      }
      setExpertToAdd('');
    }
  };

  const handleRemoveExpert = (expertId: string) => {
    setSelectedExperts(selectedExperts.filter(e => e.title !== expertId));
    const newFiles = { ...files };
    delete newFiles[expertId];
    setFiles(newFiles);
  };

  const handleFileUpload = (expertId: string, fileList: FileList | null) => {
    if (fileList) {
      setFiles(prev => ({ ...prev, [expertId]: fileList }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const newMeeting: Meeting = {
      id: Date.now().toString(),
      project_name: selectedProjectId,
      meeting_topic: title,
      rounds,
      timestamp: Number(Date.now())
    };

    const encodeFiles = async (): Promise<string[][]> => {
      const result: string[][] = [];
      for (const expert of selectedExperts) {
        const fileList = files[expert.title];
        if (!fileList || fileList.length === 0) {
          result.push([]);
          continue;
        }

        const fileArray: string[] = [];
        for (const file of Array.from(fileList)) {
          const base64 = await file.arrayBuffer().then(buffer =>
            btoa(String.fromCharCode(...new Uint8Array(buffer)))
          );
          fileArray.push(base64);
        }
        result.push(fileArray);
      }
      return result;
    };

    const vectorStore = await encodeFiles();
    const meetingRequest = {
      project_name: selectedProjectId,
      experts: selectedExperts.map(e => ({
        title: e.title,
        expertise: e.expertise,
        goal: e.goal,
        role: e.role
      })),
      vector_store: vectorStore,
      meeting_topic: title,
      rounds: rounds
    };

    setMeetings([...meetings, newMeeting]);
    navigate('/meeting-stream', { state: { meetingRequest, meetingId: newMeeting.id } });
  };

  const availableExperts = experts.filter(expert => 
    !selectedExperts.find(selected => selected.title === expert.title)
  );

  return (
    <div className="max-w-6xl mx-auto">
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-foreground">Start New Meeting</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <Label htmlFor="project" className="text-foreground">Select Project</Label>
              <Select value={selectedProjectId} onValueChange={setSelectedProjectId} required>
                <SelectTrigger className="bg-input border-border text-foreground">
                  <SelectValue placeholder="Choose a project" />
                </SelectTrigger>
                <SelectContent className="bg-popover border-border">
                {Object.values(projects).map((project) => (
                  <SelectItem key={project.title} value={project.title} className="text-popover-foreground">
                    {project.title}
                  </SelectItem>
                ))}
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="title" className="text-foreground">Meeting Topic</Label>
                <Input
                  id="title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter meeting topic"
                  className="bg-input border-border text-foreground placeholder:text-muted-foreground"
                  required
                />
              </div>
              
              <div>
                <Label htmlFor="rounds" className="text-foreground">Number of Rounds</Label>
                <Input
                  id="rounds"
                  type="number"
                  min="1"
                  max="20"
                  value={rounds}
                  onChange={(e) => setRounds(Number(e.target.value))}
                  className="bg-input border-border text-foreground"
                  required
                />
              </div>
            </div>
            
            <div>
              <Label className="text-base font-semibold text-foreground">Select Experts</Label>
              
              <div className="mt-4 flex gap-2">
                <Select value={expertToAdd} onValueChange={setExpertToAdd}>
                  <SelectTrigger className="bg-input border-border text-foreground flex-1">
                    <SelectValue placeholder="Choose an expert to add" />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {availableExperts.map((expert) => (
                      <SelectItem key={expert.title} value={expert.title} className="text-popover-foreground">
                        {expert.title} - {expert.role}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button 
                  type="button" 
                  onClick={handleAddExpert}
                  disabled={!expertToAdd}
                  className="bg-primary text-primary-foreground hover:bg-primary/90"
                >
                  Add Expert
                </Button>
              </div>

              {selectedExperts.length > 0 && (
                <div className="mt-6">
                  <Table className="bg-card border-border">
                    <TableHeader>
                      <TableRow className="border-border">
                        <TableHead className="text-foreground">Title</TableHead>
                        <TableHead className="text-foreground">Role</TableHead>
                        <TableHead className="text-foreground">Expertise</TableHead>
                        <TableHead className="text-foreground">Goal</TableHead>
                        <TableHead className="text-foreground">Upload Files</TableHead>
                        <TableHead className="text-foreground">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {selectedExperts.map((expert) => (
                        <TableRow key={expert.title} className="border-border">
                          <TableCell className="text-foreground font-medium">{expert.title}</TableCell>
                          <TableCell className="text-foreground">{expert.role}</TableCell>
                          <TableCell className="text-foreground">{expert.expertise}</TableCell>
                          <TableCell className="text-foreground">{expert.goal}</TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <Input
                                type="file"
                                accept=".pdf"
                                multiple
                                onChange={(e) => handleFileUpload(expert.title, e.target.files)}
                                className="bg-input border-border text-foreground file:text-foreground text-xs"
                              />
                              <Upload className="h-4 w-4 text-muted-foreground" />
                            </div>
                            {files[expert.title] && (
                              <p className="text-xs text-muted-foreground mt-1">
                                {files[expert.title].length} file(s) selected
                              </p>
                            )}
                          </TableCell>
                          <TableCell>
                            <Button
                              type="button"
                              variant="destructive"
                              size="sm"
                              onClick={() => handleRemoveExpert(expert.title)}
                              className="h-8 w-8 p-0"
                            >
                              <X className="h-4 w-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
              
              {experts.length === 0 && (
                <p className="text-muted-foreground text-center py-8">
                  No experts available. Create experts first in Expert Manager.
                </p>
              )}
              
              {Object.keys(projects).length === 0 && (
                <p className="text-muted-foreground text-center py-8">
                  No projects available. Create a project first in Project Manager.
                </p>
              )}
            </div>
            
            <div className="flex gap-4 pt-4">
              <Button 
                type="submit" 
                className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90"
                disabled={!selectedProjectId || selectedExperts.length === 0 || !title}
              >
                Start Meeting
              </Button>
              <Button 
                type="button" 
                variant="outline" 
                onClick={() => navigate('/projects')}
                className="flex-1 border-border text-foreground hover:bg-accent"
              >
                Cancel
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default NewMeetingForm;
