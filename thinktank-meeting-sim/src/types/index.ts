
export interface Expert {
  id: string;
  title: string;
  role: string;
  expertise: string;
  goal: string;
  files?: File[];
  createdAt: Date;
  updatedAt: Date;
}

export interface Project {
  id: string;
  title: string;
  description: string;
  experts: Expert[];
  meetings: Meeting[];
  createdAt: Date;
  updatedAt: Date;
}

export interface Meeting {
  id: string;
  projectId: string;
  title: string;
  rounds: number;
  transcript?: string;
  status: 'pending' | 'running' | 'completed';
  startedAt?: Date;
  completedAt?: Date;
  createdAt: Date;
}

export interface FileUpload {
  id: string;
  name: string;
  size: number;
  type: string;
  expertId: string;
  uploadedAt: Date;
}
