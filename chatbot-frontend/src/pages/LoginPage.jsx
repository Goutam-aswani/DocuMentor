import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom'; // Make sure Link and useLocation are 
import { useAuth } from '../hooks/useAuth'; 
import { api } from '../services/api';
import { Bot } from 'lucide-react';

export default function LoginPage() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const { login } = useAuth();
    const location = useLocation();

    // This will display a success message if you were redirected from the verify page
    const successMessage = location.state?.successMessage;

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        try {
            const token = await api.login(username, password);
            login(token);
        } catch (err) {
            setError('Incorrect username or password.');
        }
    };
        
    return (
 <div className="flex items-center justify-center h-screen text-white p-4">
 <div
 className="absolute inset-0 bg-cover bg-center"
 style={{
 backgroundImage: "url('https://i.imgur.com/3pGCUmP.jpeg')",
 }}
 ></div>
            <div className="relative z-10 w-full max-w-sm p-8 space-y-6 bg-background/80 backdrop-blur-sm rounded-2xl shadow-2xl animate-fade-in-up">
                <div className="flex flex-col items-center space-y-2">
                    <div className="bg-blue-600 p-3 rounded-full">
                      <Bot className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-2xl font-bold text-center text-white">Welcome Back</h1>
                </div>

                {successMessage && <p className="text-green-400 text-sm text-center">{successMessage}</p>}

                <form className="space-y-4" onSubmit={handleSubmit}>
                    <input
                        type="text"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        placeholder="Username"
                        className="w-full px-4 py-2 text-black bg-input border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary transition-all"
                        required
                    />
                    <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Password"
                        className="w-full px-4 py-2 text-black bg-input border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary transition-all"
                        required
                    />
                    {error && <p className="text-red-400 text-sm text-center">{error}</p>}
                    {/* <button type="submit" className="w-full px-4 py-3 font-bold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-all transform hover:scale-105"> */}
                    <button type="submit" className="w-full px-4 py-3 font-bold text-primary-foreground bg-primary rounded-lg hover:bg-primary/90 transition-all transform hover:scale-105">    
                        Login
                    </button>
                </form>

                {/* This is the link to the signup page */}
                <p className="text-center text-sm text-gray-400">
                    Don't have an account?{' '}
                    <Link to="/signup" className="font-medium text-primary hover:underline">
                        Sign Up
                    </Link>
                </p>
            </div>
        </div>
    );
};
