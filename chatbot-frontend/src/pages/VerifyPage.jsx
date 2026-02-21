// src/pages/VerifyPage.jsx

import { useState, useEffect } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { api } from '../services/api';
import { MailCheck } from 'lucide-react';

export default function VerifyPage() {
    const [otp, setOtp] = useState('');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const navigate = useNavigate();
    const location = useLocation();
    
    // Get the email from the navigation state passed from the signup page
    const email = location.state?.email;

    // If the user lands here without an email, redirect them to signup
    useEffect(() => {
        if (!email) {
            navigate('/signup');
        }
    }, [email, navigate]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');
        try {
            const response = await api.verifyEmail(email, otp);
            setSuccess(response.message);
            // Redirect to login after a short delay
            setTimeout(() => {
                navigate('/login', { state: { successMessage: 'Verification successful! You can now log in.' } });
            }, 2000);
        } catch (err) {
            setError(err.detail || 'An error occurred during verification.');
        }
    };

    return (
        <div className="flex items-center justify-center h-screen text-white p-4 bg-gradient-to-br from-gray-900 to-black">
            <div className="w-full max-w-sm p-8 space-y-6 bg-gray-800/70 backdrop-blur-md rounded-2xl shadow-2xl animate-fade-in-up">
                <div className="flex flex-col items-center space-y-2">
                    <div className="bg-yellow-600 p-3 rounded-full">
                      <MailCheck className="w-8 h-8 text-white" />
                    </div>
                    <h1 className="text-2xl font-bold text-center text-white">Verify Your Email</h1>
                    <p className="text-center text-gray-300 text-sm">
                        An OTP has been sent to <strong>{email}</strong>. Please enter it below.
                    </p>
                </div>
                <form className="space-y-4" onSubmit={handleSubmit}>
                    <input
                        type="text"
                        value={otp}
                        onChange={(e) => setOtp(e.target.value)}
                        placeholder="Enter 6-digit OTP"
                        maxLength="6"
                        className="w-full px-4 py-3 text-white text-center tracking-[0.5em] bg-gray-700 border border-transparent rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500 transition-all placeholder-gray-400"
                        required
                    />
                    {error && <p className="text-red-400 text-sm text-center">{error}</p>}
                    {success && <p className="text-green-400 text-sm text-center">{success}</p>}
                    <button type="submit" className="w-full px-4 py-3 font-bold text-white bg-yellow-600 rounded-lg hover:bg-yellow-700 transition-all transform hover:scale-105">
                        Verify
                    </button>
                </form>
                 <p className="text-center text-sm text-gray-400">
                    Didn't get an email?{' '}
                    <Link to="/signup" className="font-medium text-yellow-400 hover:underline">
                        Try signing up again
                    </Link>
                </p>
            </div>
        </div>
    );
};
