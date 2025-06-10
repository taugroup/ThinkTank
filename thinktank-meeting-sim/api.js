// api.js

const BASE_URL = 'http://localhost:8000'

// GET function to fetch all projects
export const getProjects = async () => {
    try {
        const response = await fetch(`${BASE_URL}/projects`);
        if (!response.ok) {
            throw new Error('Failed to fetch projects');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching projects:', error);
        return [];
    }
};

// POST function to add a new project
export const addProject = async (project) => {
    try{
        const response = await fetch(`${BASE_URL}/projects`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(project),
        });
        if (!response.ok) {
            throw new Error('Failed to run simulation step');
          }
          return await response.json();
    } catch (error) {
        console.error('Error adding project:', error);
        return { success: false };
    }
}


// GET function to get a specific project by name
export const getProjectByName = async (name) => {
    try {
        const response = await fetch(`${BASE_URL}/projects/${encodeURIComponent(name)}`);
        if (!response.ok) {
            throw new Error('Failed to fetch project');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching project:', error);
        return null;
    }
};

// GET all expert templates
export const getExpertTemplates = async () => {
    try {
        const response = await fetch(`${BASE_URL}/templates`);
        if (!response.ok) {
            throw new Error('Failed to fetch expert templates');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching expert templates:', error);
        return [];
    }
};

// POST function to add a new expert template
export const addExpertTemplate = async (template) => {
    try {
        const response = await fetch(`${BASE_URL}/templates`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(template),
        });
        if (!response.ok) {
            throw new Error('Failed to add expert template');
        }
        return await response.json();
    } catch (error) {
        console.error('Error adding expert template:', error);
        return { success: false };
    }
};

// Delete function to remove an expert template by name
export const deleteExpertTemplate = async (name) => {
    try {
        const response = await fetch(`${BASE_URL}/templates/${encodeURIComponent(name)}`, {
            method: 'DELETE',
        });
        if (!response.ok) {
            throw new Error('Failed to delete expert template');
        }
        return await response.json();
    } catch (error) {
        console.error('Error deleting expert template:', error);
        return { success: false };
    }
};