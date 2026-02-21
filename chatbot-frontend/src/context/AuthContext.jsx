// src/contexts/AuthContext.jsx

import { createContext, useState, useEffect } from 'react';
import { setupAuthInterceptor } from '../services/api';

// Export the context itself so the hook can use it
export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [token, setToken] = useState(localStorage.getItem('authToken'));

    const login = (newToken) => {
        localStorage.setItem('authToken', newToken);
        setToken(newToken);
    };

    const logout = () => {
        localStorage.removeItem('authToken');
        setToken(null);
    };

    useEffect(() => {
        setupAuthInterceptor(logout);
    }, []);

    return (
        <AuthContext.Provider value={{ token, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
};