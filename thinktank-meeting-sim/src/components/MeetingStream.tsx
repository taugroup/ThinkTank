import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { useLocation, useNavigate } from 'react-router-dom';

const MeetingStream = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { meetingRequest, meetingId } = location.state || {};
    const [messages, setMessages] = useState<{ name: string; content: string }[]>([]);
  
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(true);
  
    useEffect(() => {
      if (!meetingRequest) {
        // No meeting data passed, redirect back
        navigate('/new-meeting');
        return;
      }
  
      const socket = new WebSocket('ws://localhost:8000/ws/meeting'); // or wss://
  
      socket.onopen = () => {
        console.log('WebSocket connected');
        setLoading(true);
        socket.send(JSON.stringify(meetingRequest));
      };
  
      socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            setMessages(prev => [...prev, data]);
          } catch (err) {
            console.error('Error parsing WS message JSON', err);
          }
      };
  
      socket.onerror = (err) => {
        console.error('WebSocket error:', err);
        setLoading(false);
      };
  
      socket.onclose = () => {
        console.log('WebSocket closed');
        setLoading(false);
      };
  
      return () => {
        socket.close();
      };
    }, [meetingRequest, navigate]);
    return (
        <div className="max-w-4xl mx-auto p-4">
          <h1 className="text-2xl font-bold mb-4">Meeting Response Stream</h1>
          {loading && <p>Loading response...</p>}
          
          <div className="space-y-6 bg-gray-100 p-4 rounded max-h-[60vh] overflow-y-auto">
            {messages.map((msg, idx) => (
                <div key={idx} className="border-b border-gray-300 pb-4">
                <ReactMarkdown className="prose prose-sm">
                    {`${msg.name}\n\n${msg.content}`}
                </ReactMarkdown>
                </div>
            ))}
            </div>
      
          {!loading && (
            <button
              onClick={() => navigate('/projects')}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded"
            >
              Back to Projects
            </button>
          )}
        </div>
      );
  };
  
  export default MeetingStream;