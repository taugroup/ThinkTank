
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { Expert } from '@/types';

const NewExpertForm = () => {
  const navigate = useNavigate();
  const [experts, setExperts] = useLocalStorage<Expert[]>('experts', []);
  const [title, setTitle] = useState('');
  const [role, setRole] = useState('');
  const [expertise, setExpertise] = useState('');
  const [goal, setGoal] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const newExpert: Expert = {
      id: Date.now().toString(),
      title,
      role,
      expertise,
      goal,
      files: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    setExperts([...experts, newExpert]);
    navigate('/experts');
  };

  return (
    <div className="max-w-2xl mx-auto">
      <Card>
        <CardHeader>
          <CardTitle>Create New Expert</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <Label htmlFor="title">Expert Title</Label>
              <Input
                id="title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Enter expert title"
                required
              />
            </div>
            
            <div>
              <Label htmlFor="role">Role</Label>
              <Input
                id="role"
                value={role}
                onChange={(e) => setRole(e.target.value)}
                placeholder="Enter expert role"
                required
              />
            </div>
            
            <div>
              <Label htmlFor="expertise">Expertise</Label>
              <Textarea
                id="expertise"
                value={expertise}
                onChange={(e) => setExpertise(e.target.value)}
                placeholder="Enter expert expertise"
                rows={3}
                required
              />
            </div>
            
            <div>
              <Label htmlFor="goal">Goal</Label>
              <Textarea
                id="goal"
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                placeholder="Enter expert goal"
                rows={3}
                required
              />
            </div>
            
            <div className="flex gap-4 pt-4">
              <Button type="submit" className="flex-1">
                Create Expert
              </Button>
              <Button 
                type="button" 
                variant="outline" 
                onClick={() => navigate('/experts')}
                className="flex-1"
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

export default NewExpertForm;
