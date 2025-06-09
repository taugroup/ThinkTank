
export interface Expert {
  title: string;
  role: string;
  expertise: string;
  goal: string;
}

export interface Project {
  title: string;
  description: string;
  meetings?: Meeting[];
}

export interface Meeting {
  id: string;
  projectTitle: string;
  topic: string;
  timestamp: BigInt
  rounds: number;
  transcript?: string;
  summary?: string;
  experts?: Expert[];
}

export interface FileUpload {
  id: string;
  name: string;
  size: number;
  type: string;
  expertId: string;
  uploadedAt: Date;
}
