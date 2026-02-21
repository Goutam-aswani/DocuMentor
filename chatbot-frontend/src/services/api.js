// src/services/api.js

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const apiClient = axios.create({
    baseURL: API_BASE_URL,
});

export const setupAuthInterceptor = (logout) => {
    apiClient.interceptors.response.use(
        (response) => response,
        (error) => {
            if (error.response && error.response.status === 401) {
                logout();
            }
            return Promise.reject(error);
        }
    );
};


// CHANGE: We are adding a new, separate function specifically for streaming.
// This uses the native Fetch API because it handles response streams more effectively than Axios.
/**
 * Posts a message and handles the streaming response.
 * @param {string} token - The user's authentication token.
 * @param {string} prompt - The user's message.
 * @param {number|null} sessionId - The ID of the current chat session.
 * @param {string} modelName - The selected model name.
 * @param {boolean} useWebSearch - Whether to use web search.
 * @param {function(string): void} onChunk - A callback function to handle each received chunk of text.
 * @returns {Promise<void>} A promise that resolves when the stream is complete.
 */
export async function streamMessage(token, prompt, sessionId, modelName = "llama-3.3-70b-versatile", useWebSearch = false, onChunk) {
    try {
        // DEBUG: Log the start of the fetch request
        const payload = { 
            prompt, 
            session_id: sessionId,  // This will be null for new conversations
            model_name: modelName,
            use_web_search: useWebSearch
        };
        console.log(`--- DEBUG: Starting stream fetch for session ${sessionId || 'new'} with model ${modelName} ---`);
        console.log(`--- DEBUG: Web search enabled: ${useWebSearch} ---`);
        
        const response = await fetch(`${API_BASE_URL}/chats/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            // Handle HTTP errors like 404 or 500
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        if (!response.body) {
            throw new Error("Response body is null");
        }
        
        const sessionIdFromHeaders = response.headers.get('X-Session-ID');
        const sessionWasCreated = response.headers.get('X-Session-Created') === 'true';
        
        // *** ENHANCED DEBUG: Log session information from headers ***
        console.log('--- DEBUG: Session ID from headers:', sessionIdFromHeaders);
        console.log('--- DEBUG: Session was created:', sessionWasCreated);
        // The reader allows us to process the response stream chunk by chunk.
        const reader = response.body.getReader();
        // The decoder converts the raw binary chunks (Uint8Array) into readable text.
        const decoder = new TextDecoder();
        
        // DEBUG: Log that the frontend is ready to read the stream.
        console.log("--- DEBUG: Frontend ready to read stream ---");

        // Loop indefinitely to read from the stream
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                // DEBUG: Log when the stream has ended.
                console.log("--- DEBUG: Stream finished ---");
                break; // Exit the loop when the stream is complete.
            }
            
            const chunk = decoder.decode(value);
            // DEBUG: Log each chunk received from the backend.
            console.log("--- DEBUG: FRONTEND CHUNK:", chunk);
            
            // Call the provided callback function with the new chunk of text.
            onChunk(chunk);
        }
        return {
            sessionId: sessionIdFromHeaders ? parseInt(sessionIdFromHeaders) : sessionId,
            sessionWasCreated: sessionWasCreated
            // originalSessionId: sessionId
        };

    } catch (error) {
        // DEBUG: Log any errors that occur during the streaming process.
        console.error("--- DEBUG: Error during streaming:", error);
        // Re-throw the error so it can be caught by the component.
        throw error;
    }
}


export const api = {
    
    async register(username, email, password) {
        try {
            const payload = { username, email, password };
            const response = await apiClient.post('/users/register', payload);
            return response.data;
        } catch (error) {
            console.error('Registration failed:', error.response?.data || error.message);
            throw error.response?.data || new Error('Registration failed');
        }
    },

    async verifyEmail(email, otp) {
        try {
            const payload = { email, otp };
            const response = await apiClient.post('/users/verify-email', payload);
            return response.data;
        } catch (error) {
            console.error('Email verification failed:', error.response?.data || error.message);
            throw error.response?.data || new Error('Email verification failed');
        }
    },

    async login(username, password) {
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);
        try {
            const response = await apiClient.post('/auth/token', formData);
            return response.data.access_token;
        } catch (error) {
            console.error('Login failed:', error.response?.data || error.message);
            throw new Error('Login failed');
        }
    },

    async getChatSessions(token) {
        try {
            const response = await apiClient.get('/chats/', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to fetch sessions:', error.response?.data || error.message);
            throw new Error('Failed to fetch sessions');
        }
    },

    async getChatHistory(token, sessionId) {
        try {
            const response = await apiClient.get(`/chats/${sessionId}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to fetch history:', error.response?.data || error.message);
            throw new Error('Failed to fetch history');
        }
    },

    // NOTE: This function is no longer used for sending messages but is kept for reference.
    // The new `streamMessage` function above is now used instead.
    async uploadDocument(token, sessionId, file) {
        const formData = new FormData();
        formData.append('file', file);
    
        const response = await fetch(`${API_BASE_URL}/rag/upload/${sessionId}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
            },
            body: formData,
        });
    
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to upload document');
        }
    
        return await response.json();
    },

    async getAvailableModels(token) {
        try {
            const response = await apiClient.get('/chats/models', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to fetch models:', error.response?.data || error.message);
            throw new Error('Failed to fetch models');
        }
    },

    async postMessage(token, prompt, sessionId) {
        try {
            const payload = { prompt, session_id: sessionId };
            const response = await apiClient.post('/chats/', payload, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                },
            });
            return response.data;
        } catch (error) {
            console.error('Failed to post message:', error.response?.data || error.message);
            throw new Error('Failed to post message');
        }
    },
    async deleteChatSession(token, sessionId) {
        try {
            const response = await apiClient.delete(`/chats/${sessionId}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to delete session:', error.response?.data || error.message);
            throw new Error('Failed to delete session');
        }
    },
    async renameChatSession(token, sessionId, newTitle) {
        try {
            const payload = { new_title: newTitle };
            const response = await apiClient.put(`/chats/${sessionId}`, payload, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json',
                },
            });
            return response.data;
        } catch (error) {
            console.error('Failed to rename session:', error.response?.data || error.message);
            throw new Error('Failed to rename session');
        }
    },
    async getUserProfile(token) {
        try {
            const response = await apiClient.get('/users/me', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to fetch user profile:', error.response?.data || error.message);
            throw new Error('Failed to fetch user profile');
        }
    },
    async updateUserProfile(token, profileData) {
        try {
            const response = await apiClient.put('/users/me', profileData, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to update user profile:', error.response?.data || error.message);
            throw new Error('Failed to update user profile');
        }
    },
    async forgotPassword(email) {
        try {
            const payload = { email };
            const response = await apiClient.post('/users/forget_password', payload);
            return response.data;
        } catch (error) {
            console.error('Forgot password request failed:', error.response?.data || error.message);
            throw error.response?.data || new Error('Forgot password request failed');
        }
    },


    // async changePassword(token, currentPassword, newPassword) {
    //     try {
    //         const payload = { current_password: currentPassword, new_password: newPassword };
    //         const response = await apiClient.put('/users/change-password', payload, {
    //             headers: { 'Authorization': `Bearer ${token}` }
    //         });
    //         return response.data;
    //     } catch (error) {
    //         console.error('Failed to change password:', error.response?.data || error.message);
    //         throw new Error('Failed to change password');
    //     }
    // }

    // Usage Statistics API
    async getUsageStats(token, days = 30) {
        try {
            const response = await apiClient.get(`/chats/usage/stats?days=${days}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to fetch usage stats:', error.response?.data || error.message);
            throw new Error('Failed to fetch usage statistics');
        }
    },

    async getUsageTotals(token) {
        try {
            const response = await apiClient.get('/chats/usage/totals', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            return response.data;
        } catch (error) {
            console.error('Failed to fetch usage totals:', error.response?.data || error.message);
            throw new Error('Failed to fetch usage totals');
        }
    }
};
