// src/pages/ProfilePage.jsx

import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { api } from '../services/api';
import { UserCircle, ArrowLeft } from 'lucide-react';

export default function ProfilePage() {
    const { token } = useAuth();
    const [profile, setProfile] = useState(null);
    const [fullName, setFullName] = useState('');
    const [email, setEmail] = useState('');
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    useEffect(() => {
        const fetchProfile = async () => {
            try {
                const data = await api.getUserProfile(token);
                setProfile(data);
                setFullName(data.full_name || '');
                setEmail(data.email || '');
            } catch (err) {
                setError('Failed to load profile.');
            } finally {
                setIsLoading(false);
            }
        };
        fetchProfile();
    }, [token]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');
        setIsLoading(true);
        try {
            const updatedProfile = await api.updateUserProfile(token, {
                full_name: fullName,
                email: email,
            });
            setProfile(updatedProfile);
            setSuccess('Profile updated successfully!');
        } catch (err) {
            setError(err.detail || 'Failed to update profile.');
        } finally {
            setIsLoading(false);
        }
    };

    if (isLoading && !profile) {
        return <div className="flex items-center justify-center h-screen bg-gray-900 text-white">Loading...</div>;
    }

    return (
        <div className="flex items-center justify-center h-screen bg-gray-900 text-white p-4">
            <div className="w-full max-w-md p-8 space-y-6 bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-2xl">
                <div className="flex flex-col items-center space-y-2">
                    <div className="bg-blue-600 p-4 rounded-full">
                        <UserCircle className="w-12 h-12 text-white" />
                    </div>
                    <h1 className="text-2xl font-bold text-center text-white">Your Profile</h1>
                </div>

                <form className="space-y-4" onSubmit={handleSubmit}>
                    <div>
                        <label className="text-sm font-medium text-gray-400">Username</label>
                        <input
                            type="text"
                            value={profile?.username || ''}
                            className="w-full px-4 py-2 mt-1 text-gray-300 bg-gray-700 border border-gray-600 rounded-lg cursor-not-allowed"
                            readOnly
                            disabled
                        />
                    </div>
                    <div>
                        <label htmlFor="fullName" className="text-sm font-medium text-gray-400">Full Name</label>
                        <input
                            id="fullName"
                            type="text"
                            value={fullName}
                            onChange={(e) => setFullName(e.target.value)}
                            placeholder="Your full name"
                            className="w-full px-4 py-2 mt-1 text-white bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                    </div>
                    <div>
                        <label htmlFor="email" className="text-sm font-medium text-gray-400">Email</label>
                        <input
                            id="email"
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="Your email address"
                            className="w-full px-4 py-2 mt-1 text-white bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                            required
                        />
                    </div>

                    {error && <p className="text-red-400 text-sm text-center">{error}</p>}
                    {success && <p className="text-green-400 text-sm text-center">{success}</p>}

                    <button type="submit" disabled={isLoading} className="w-full px-4 py-3 font-bold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-all transform hover:scale-105 disabled:bg-gray-500">
                        {isLoading ? 'Saving...' : 'Save Changes'}
                    </button>
                </form>

                <div className="text-center">
                    <Link to="/chat" className="inline-flex items-center text-sm text-blue-400 hover:underline">
                        <ArrowLeft className="w-4 h-4 mr-2" />
                        Back to Chat
                    </Link>
                </div>
            </div>
        </div>
    );
}