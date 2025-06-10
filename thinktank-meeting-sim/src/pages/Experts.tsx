
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useEffect, useState } from "react";
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Plus, Edit, Trash2 } from 'lucide-react';
import { useLocalStorage } from '@/hooks/useLocalStorage';
import { Expert } from '@/types';
import { getExpertTemplates, deleteExpertTemplate } from '../../api'
import { get } from 'http';

const Experts = () => {
  const navigate = useNavigate();
  const [experts, setExperts] = useState<Expert[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchExperts = async () => {
      try {
        const rawExperts = await getExpertTemplates();
        setExperts(rawExperts as Expert[]);
      } catch (err) {
        console.error("Failed to fetch projects", err);
      } finally {
        setLoading(false);
      }
    };
    fetchExperts();
  }, []);

  const handleDelete = (expertTitle: string) => {
    if (confirm('Are you sure you want to delete this expert?')) {
      deleteExpertTemplate(expertTitle).catch(error => {
        console.error("Error deleting expert template:", error);
        // Optionally, show an error message to the user
      });
      setExperts(experts.filter(expert => expert.title !== expertTitle));
    }
  };

  const handleUpdateExpert = (expert: Expert) => {
    navigate('/experts/new', { state: { expert } });
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-semibold">Expert Manager</h1>
          <p className="text-muted-foreground">Manage your expert templates</p>
        </div>
        <Link to="/experts/new">
          <Button className="flex items-center gap-2">
            <Plus className="h-4 w-4" />
            New Expert
          </Button>
        </Link>
      </div>

      {experts.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-muted-foreground mb-4">No experts yet</p>
          <Link to="/experts/new">
            <Button>Create your first expert</Button>
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {experts.map((expert) => (
            <div key={expert.title} className="border rounded-lg p-4 space-y-3">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-medium">{expert.title}</h3>
                  <p className="text-sm text-muted-foreground">{expert.role}</p>
                </div>
                <div className="flex gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => handleUpdateExpert(expert)}
                  >
                    <Edit className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => handleDelete(expert.title)}
                    className="h-8 w-8 text-destructive hover:text-destructive"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <div>
                <p className="text-sm font-medium">Expertise:</p>
                <p className="text-sm text-muted-foreground">{expert.expertise}</p>
              </div>
              <div>
                <p className="text-sm font-medium">Goal:</p>
                <p className="text-sm text-muted-foreground">{expert.goal}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Experts;
