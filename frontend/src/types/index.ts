
export interface FileData {
  filename: string;
  content: string; // base64-encoded file content
}

export interface FileReference {
  original_name: string;
  size: number;
}

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
  project_name: string;
  meeting_topic: string;
  timestamp: number
  rounds: number;
  transcript?: [{ name: string; content: string }];
  summary?: string;
  experts?: Expert[];
}

export interface FileUpload {
  expertId: string;
  files: string[];
}
