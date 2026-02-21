// src/hooks/useAuth.js

import { useContext } from 'react';
import { AuthContext } from '../context/AuthContext';

// The hook now lives in its own file
export const useAuth = () => useContext(AuthContext);
