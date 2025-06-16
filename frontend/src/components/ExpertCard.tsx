
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Edit, Trash2, Upload, FileText } from 'lucide-react';
import { Expert } from '@/types';

interface ExpertCardProps {
  expert: Expert;
  onEdit: (expert: Expert) => void;
  onDelete: (expertId: string) => void;
  onUploadFiles: (expertId: string) => void;
}

const ExpertCard: React.FC<ExpertCardProps> = ({ 
  expert, 
  onEdit, 
  onDelete
}) => {
  return (
    <Card className="hover:shadow-lg transition-all duration-300 animate-fade-in">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-lg font-semibold mb-1">{expert.title}</CardTitle>
            <Badge variant="outline" className="mb-2">{expert.role}</Badge>
          </div>
          <div className="flex space-x-1">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => onEdit(expert)}
              className="h-8 w-8"
            >
              <Edit className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => onDelete(expert.title)}
              className="h-8 w-8 text-destructive hover:text-destructive"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="pt-0">
        <div className="space-y-3">
          <div>
            <h4 className="text-sm font-medium text-foreground mb-1">Expertise</h4>
            <p className="text-sm text-muted-foreground">{expert.expertise}</p>
          </div>
          
          <div>
            <h4 className="text-sm font-medium text-foreground mb-1">Goal</h4>
            <p className="text-sm text-muted-foreground">{expert.goal}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ExpertCard;
