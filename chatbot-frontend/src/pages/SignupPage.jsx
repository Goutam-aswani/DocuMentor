// src/pages/SignupPage.jsx

import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import { UserPlus } from 'lucide-react';

export default function SignupPage() {
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        try {
            await api.register(username, email, password);
            // On success, navigate to the verification page, passing the email along
            navigate('/verify', { state: { email } });
        } catch (err) {
            setError(err.detail || 'An error occurred during registration.');
        }
    };

    return (<div
    className="flex items-center justify-center h-screen text-white p-4 bg-cover bg-center relative"
    style={{
      backgroundImage: "url('https://i.imgur.com/GoeKoPX.jpeg')" // put your image inside /public folder
    }}
  >
            <div className="w-full max-w-sm p-8 space-y-6 bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-2xl">
                <div className="flex flex-col items-center space-y-2">
                    <div className="bg-green-600 p-3 rounded-full">
                      <UserPlus className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-2xl font-bold text-center text-white">Create Account</h1>
                </div>
                <form className="space-y-4" onSubmit={handleSubmit}>
                    <input
                        type="text"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        placeholder="Username"
                        className="w-full px-4 py-2 text-foreground bg-input border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary transition-all"
                        required
                    />
                    <input
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="Email"
                        className="w-full px-4 py-2 text-foreground bg-input border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary transition-all"
                        required
                    />
                    <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Password"
                        className="w-full px-4 py-2 text-foreground bg-input border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary transition-all"
                        required
                    />
                    {error && <p className="text-red-400 text-sm text-center">{error}</p>}
                    <button type="submit" className="w-full px-4 py-3 font-bold text-primary-foreground bg-primary rounded-lg hover:opacity-90 transition-opacity transform hover:scale-105">
                        Sign Up
                    </button>
                </form>
                <p className="text-center text-sm text-gray-400">
                    Already have an account?{' '}
                    <Link to="/login" className="font-medium text-blue-400 hover:underline">
                        Log In
                    </Link>
                </p>
            </div>
        </div>
    );
};
