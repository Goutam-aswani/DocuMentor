// /chatbot-frontend/src/pages/ChatPage.jsx - Updated handleSendMessage function with typing delay

import { useState, useEffect, useRef } from 'react';
import { useAuth } from '../hooks/useAuth';
import { api, streamMessage } from '../services/api';
import Sidebar from '../components/Sidebar';
import ChatWindow from '../components/ChatWindow';
import ChatSettings from '../components/ChatSettings';
import KeyboardShortcutsModal from '../components/KeyboardShortcutsModal';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';
import { Hand, PanelRight, Paperclip } from 'lucide-react';

export default function ChatPage() {
    const { token } = useAuth();
    const [sessions, setSessions] = useState([]);
    const [activeSession, setActiveSession] = useState(null);
    const [currentUser,SetCurrentUser] = useState(null);
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isHistoryLoading, setIsHistoryLoading] = useState(false);
    const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
    const [availableModels, setAvailableModels] = useState({});
    const [selectedModel, setSelectedModel] = useState('llama-3.3-70b-versatile');
    const [useWebSearch, setUseWebSearch] = useState(false);
    const [isModelsLoading, setIsModelsLoading] = useState(false);
    const [isShortcutsModalOpen, setIsShortcutsModalOpen] = useState(false);
    const fileInputRef = useRef(null);
    const chatInputRef = useRef(null);

    // Ref to keep track of the ongoing typing animation
    const typingTimeoutRef = useRef(null);

    const fetchSessions = async () => {
        if (!token) return;
        setIsHistoryLoading(true);
        try {
            const data = await api.getChatSessions(token);
            setSessions(data);

            console.log('--- DEBUG: Fetched data:', data);
            console.log('--- DEBUG: Fetched sessions:', sessions);
        } catch (error) {
            console.error("Failed to fetch sessions:", error);
        } finally {
            setIsHistoryLoading(false);
        }
    };

    const fetchAvailableModels = async () => {
        if (!token) return;
        setIsModelsLoading(true);
        try {
            const models = await api.getAvailableModels(token);
            setAvailableModels(models);
        } catch (error) {
            console.error("Failed to fetch models:", error);
        } finally {
            setIsModelsLoading(false);
        }
    };

    useEffect(() => {
        fetchSessions();
        fetchAvailableModels();
    }, [token]);

    // Define handler functions first
    const handleNewChat = () => {
        setActiveSession(null);
        setMessages([]);
    };

    const handleSelectSession = async (sessionId) => {
        setIsLoading(true);
        setActiveSession(sessionId);
        try {
            const data = await api.getChatHistory(token, sessionId);
            setMessages(data.messages);
        } catch (error) {
            console.error("Failed to fetch chat history:", error);
            setMessages([]);
        } finally {
            setIsLoading(false);
        }
    };

    // Keyboard shortcuts handlers
    const handleFocusInput = () => {
        if (chatInputRef.current) {
            chatInputRef.current.focus();
        }
    };

    const handleToggleSidebar = () => {
        setIsSidebarCollapsed(prev => !prev);
    };

    const handleToggleSettings = () => {
        // You can implement a settings modal later if needed
        console.log('Settings shortcut pressed');
    };

    const handleClearChat = () => {
        if (messages.length > 0 && window.confirm('Are you sure you want to clear this chat?')) {
            handleNewChat();
        }
    };

    // Initialize keyboard shortcuts
    useKeyboardShortcuts({
        onNewChat: handleNewChat,
        onFocusInput: handleFocusInput,
        onToggleSidebar: handleToggleSidebar,
        onToggleSettings: handleToggleSettings,
        onClearChat: handleClearChat,
        inputRef: chatInputRef
    });

    // Handle '?' key for shortcuts modal
    useEffect(() => {
        const handleKeyPress = (event) => {
            if (event.key === '?' && !event.ctrlKey && !event.metaKey) {
                const isInputActive = event.target.tagName === 'INPUT' || 
                                     event.target.tagName === 'TEXTAREA' || 
                                     event.target.contentEditable === 'true';
                if (!isInputActive) {
                    event.preventDefault();
                    setIsShortcutsModalOpen(true);
                }
            }
        };

        document.addEventListener('keydown', handleKeyPress);
        return () => document.removeEventListener('keydown', handleKeyPress);
    }, []);

    // Cleanup the typing timeout when the component unmounts or a new message is sent
    useEffect(() => {
        return () => {
            if (typingTimeoutRef.current) {
                clearTimeout(typingTimeoutRef.current);
            }
        };
    }, [messages]); // Add messages as a dependency to clear timeout when new message is added


    const handleSendMessage = async (prompt) => {
        console.log('--- DEBUG: handleSendMessage called with activeSession:', activeSession);

        const optimisticUserMessage = { id: `user-${Date.now()}`, role: 'user', content: prompt };
        // Add a unique ID for the bot placeholder message to help with updates
        const botPlaceholder = { id: `model-${Date.now()}`, role: 'model', content: '' };

        setMessages(prev => [...prev, optimisticUserMessage, botPlaceholder]);
        setIsLoading(true);

        try {
            // Store the accumulated response outside the onChunk for processing
            let accumulatedResponse = '';

            const streamResponse = await streamMessage(
                token,
                prompt,
                activeSession,
                selectedModel,
                useWebSearch,
                (chunk) => {
                    // Append the new chunk to the accumulated response
                    accumulatedResponse += chunk;

                    // Clear any existing typing timeout
                    if (typingTimeoutRef.current) {
                        clearTimeout(typingTimeoutRef.current);
                    }

                    // Start a new typing animation for the accumulated response
                    typeResponse(botPlaceholder.id, accumulatedResponse);
                }
            );

            if (streamResponse && streamResponse.sessionWasCreated) {
                console.log('--- DEBUG: New session was created, updating activeSession to:', streamResponse.sessionId);
                setActiveSession(streamResponse.sessionId);
            } else if (streamResponse && streamResponse.sessionId && !activeSession) {
                console.log('--- DEBUG: Received session ID, setting activeSession to:', streamResponse.sessionId);
                setActiveSession(streamResponse.sessionId);
            }

        } catch (error) {
            console.error("--- DEBUG: Failed to stream message:", error);
            setMessages(prev => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last && last.role === 'model') {
                    last.content = 'Error: Could not get a response from the server.';
                }
                return updated;
            });
        } finally {
            setIsLoading(false);
            fetchSessions();
        }
    };

    // Function to simulate typing effect
    const typeResponse = (messageId, textToType) => {
        setMessages(prevMessages => {
            const updatedMessages = [...prevMessages];
            const messageIndex = updatedMessages.findIndex(msg => msg.id === messageId);

            if (messageIndex !== -1) {
                const currentContent = updatedMessages[messageIndex].content;
                const charactersToType = textToType.substring(currentContent.length);

                if (charactersToType.length > 0) {
                    const nextChar = charactersToType[0];
                    updatedMessages[messageIndex].content += nextChar;

                    // Schedule the next character with a delay
                    typingTimeoutRef.current = setTimeout(() => {
                        typeResponse(messageId, textToType);
                    }, 50); // Adjust the delay (milliseconds) here for typing speed
                } else {
                     // If all characters are typed, clear the timeout
                     if (typingTimeoutRef.current) {
                        clearTimeout(typingTimeoutRef.current);
                    }
                }
            }
            return updatedMessages;
        });
    };


    const handleDeleteSession = async (sessionId) => {
        try {
            await api.deleteChatSession(token, sessionId);
            setSessions(prev => prev.filter(s => s.id !== sessionId));
            if (activeSession === sessionId) {
                handleNewChat();
            }
        } catch (error) {
            console.error("Failed to delete session:", error);
        }
    };

    const handleRenameSession = async (sessionId, newTitle) => {
        try {
            const updatedSession = await api.renameChatSession(token, sessionId, newTitle);
            setSessions(prev => prev.map(s => s.id === sessionId ? updatedSession : s));
        } catch (error) {
            console.error("Failed to rename session:", error);
        }
    };
    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        if (!activeSession) {
            alert("Please start a conversation before uploading a document.");
            return;
        }

        console.log(`Uploading ${file.name} to session ${activeSession}...`);

        try {
            const response = await api.uploadDocument(token, activeSession, file);
            alert(`Successfully uploaded ${file.name}!`);
            console.log(response.message);
        } catch (error) {
            console.error("File upload failed:", error);
            alert("Failed to upload the document. Please try again.");
        } finally {
            event.target.value = null;
        }
    };

    return (
        <div className="flex h-screen font-sans overflow-hidden">
            <Sidebar
                isCollapsed={isSidebarCollapsed}
                onToggleCollapse={() => setIsSidebarCollapsed(prev => !prev)}
                sessions={sessions}
                onSelectSession={handleSelectSession}
                onNewChat={handleNewChat}
                onDeleteSession={handleDeleteSession}
                onRenameSession={handleRenameSession}
                activeSessionId={activeSession}
                isLoading={isHistoryLoading}
            />
            <div className="flex flex-col flex-1 min-h-0">
                <ChatSettings
                    selectedModel={selectedModel}
                    onModelChange={setSelectedModel}
                    useWebSearch={useWebSearch}
                    onWebSearchToggle={setUseWebSearch}
                    availableModels={availableModels}
                    isLoading={isLoading || isModelsLoading}
                />
                <ChatWindow
                    messages={messages}
                    onSendMessage={handleSendMessage}
                    isLoading={isLoading}
                    onFileUpload={handleFileUpload}
                    fileInputRef={fileInputRef}
                    chatInputRef={chatInputRef}
                />
            </div>
            
            {/* Keyboard Shortcuts Modal */}
            <KeyboardShortcutsModal
                isOpen={isShortcutsModalOpen}
                onClose={() => setIsShortcutsModalOpen(false)}
            />
        </div>
    );
};
