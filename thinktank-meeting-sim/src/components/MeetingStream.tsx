import React, { useEffect, useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { useLocation, useNavigate } from 'react-router-dom';

const MeetingStream = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const meetingRequestRef = useRef(location.state?.meetingRequest);
    const [messages, setMessages] = useState<{ name: string; content: string }[]>([]);
    const [status, setStatus] = useState('Initializing...');
    const socketRef = useRef<WebSocket | null>(null);
  
    useEffect(() => {
        if (!meetingRequestRef.current) {
            navigate('/projects/new');
            return;
            }
        if (socketRef.current) {
            return;
            }
  
        const socket = new WebSocket('ws://localhost:8000/ws/meeting'); // or wss://
        socketRef.current = socket;
        setStatus('Connecting...');

        socket.onopen = () => {
            console.log('WebSocket connection established.');
            setStatus('Connection open. Starting meeting...');
            // Send the payload now that the connection is confirmed open.
            socket.send(JSON.stringify(meetingRequestRef.current));
          };
    
        socket.onmessage = (event) => {
        // This will now reliably receive the first message.
        console.log('WebSocket message received:', event.data);
        setStatus('Receiving data...');
        try {
            const data = JSON.parse(event.data);
            if (data.name === '__end__') {
                setStatus('Meeting complete.');
                socket.close();
            } else {
                setMessages(prev => [...prev, data]);
            }
            } catch (err) {
            console.error('Error parsing WS message JSON', err);
            }
        };
    
        socket.onerror = (err) => {
            console.error('WebSocket error:', err);
            setStatus('Error occurred. Please try again later.');
        };
    
        socket.onclose = (event) => {
            console.log('WebSocket closed.', event.reason);
            setStatus('Disconnected.');
            // Clean up the ref when the socket closes
            socketRef.current = null;
          };
    
        return () => {
        if (socketRef.current) {
            console.log("Component unmounting. Closing WebSocket.");
            socketRef.current.close();
            socketRef.current = null;
        }
        };
    }, [navigate]); 
    return (
        <div className="max-w-4xl mx-auto p-4">
          <h1 className="text-2xl font-bold mb-1">Meeting In Progress</h1>
          <p className="text-sm text-gray-500 mb-4">Status: {status}</p>
          
          <div className="space-y-6 bg-gray-800 text-white p-4 rounded-lg min-h-[60vh] max-h-[75vh] overflow-y-auto">
            {messages.length === 0 && status.startsWith('Receiving') && (
              <p className="text-center text-gray-400">Waiting for first message...</p>
            )}
            {messages.map((msg, idx) => (
                <div key={idx} className="border-b border-gray-600 pb-4 last:border-b-0">
                <ReactMarkdown className="prose prose-sm prose-invert max-w-none">
                    {`## ${msg.name}\n\n${msg.content}`}
                </ReactMarkdown>
                </div>
            ))}
            </div>
      
          {status === 'Meeting complete.' && (
            <button
              onClick={() => navigate('/projects')}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Back to Projects
            </button>
          )}
        </div>
      );
  };
  
  export default MeetingStream;