import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { api } from '../services/api';

// Import components
import { MessageDisplay } from '../components/ui/MessageDisplay';
import { SettingsHeader } from '../components/settings/SettingsHeader';
import { SettingsTabs } from '../components/settings/SettingsTabs';
import { SettingsContent } from '../components/settings/SettingsContent';

export default function SettingsPage() {
  // Auth and navigation
  const { token } = useAuth();
  const navigate = useNavigate();

  // UI states
  const [activeTab, setActiveTab] = useState('account');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isProfileLoading, setIsProfileLoading] = useState(false);
  const [confirmAction, setConfirmAction] = useState(null);

  // Form states
  const [email, setEmail] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [currentPassword, setCurrentPassword] = useState('');
  const [showPasswords, setShowPasswords] = useState({
    current: false,
    new: false,
    confirm: false
  });

  // Settings state
  const [settings, setSettings] = useState({
    // Account
    full_name: '',
    email: '',
    
    // Chat
    language: 'en',
    saveHistory: true,
    autoSave: true,
    messageSound: true,
    typingIndicator: true,
    messageLayout: 'comfortable',
    
    // Notifications
    pushNotifications: true,
    emailNotifications: 'important',
    soundNotifications: true,
    desktopNotifications: true,
    
    // Privacy
    twoFactorAuth: false,
    onlineStatus: true,
    readReceipts: true,
    lastSeen: true,
    
    // Appearance
    theme: 'auto',
    fontSize: 'medium',
    chatBackground: 'default'
  });

  // Effects
  useEffect(() => {
    const fetchProfile = async () => {
      setIsProfileLoading(true);
      try {
        const profileData = await api.getUserProfile(token);
        setSettings(prev => ({
          ...prev,
          full_name: profileData.full_name || '',
          email: profileData.email || ''
        }));
      } catch (err) {
        setError('Could not load your profile data.');
      } finally {
        setIsProfileLoading(false);
      }
    };

    fetchProfile();
  }, [token]);

  useEffect(() => {
    const applyTheme = (theme) => {
      if (theme === 'dark') {
        document.body.classList.remove('light');
      } else if (theme === 'light') {
        document.body.classList.add('light');
      } else { // 'auto'
        // Check system preference
        if (window.matchMedia('(prefers-color-scheme: light)').matches) {
          document.body.classList.add('light');
        } else {
          document.body.classList.remove('light');
        }
      }
    };

    applyTheme(settings.theme);

    // Optional: Listen for system theme changes if theme is 'auto'
    const mediaQuery = window.matchMedia('(prefers-color-scheme: light)');
    const handleSystemThemeChange = () => applyTheme(settings.theme);
    mediaQuery.addEventListener('change', handleSystemThemeChange);
    return () => mediaQuery.removeEventListener('change', handleSystemThemeChange);
  }, [settings.theme]);

  // Handlers
  const handleSettingChange = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    // Add your API call here to save settings
    console.log(`Setting ${key} changed to:`, value);
  };

  const handleSaveProfile = async () => {
    setError('');
    setMessage('');
    setIsLoading(true);
    try {
      const profileData = {
        full_name: settings.full_name,
        email: settings.email
      };
      await api.updateUserProfile(token, profileData);
      setMessage('Profile updated successfully!');
    } catch (err) {
      setError(err.detail || 'Failed to update profile.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetPassword = async () => {
    if (!email) return;
    
    setError('');
    setMessage('');
    setIsLoading(true);

    try {
      // Replace with your actual API call
      // const response = await api.forgotPassword(email);
      // setMessage(response.message);
      setMessage('Password reset link sent to your email');
      setEmail('');
    } catch (err) {
      setError(err.detail || 'An error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleChangePassword = async () => {
    if (newPassword !== confirmPassword) {
      setError('New passwords do not match');
      return;
    }
    
    if (!currentPassword || !newPassword) {
      setError('Please fill in all password fields');
      return;
    }
    
    setError('');
    setMessage('');
    setIsLoading(true);

    try {
      // Replace with your actual API call
      // await api.changePassword(currentPassword, newPassword);
      setMessage('Password changed successfully');
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err) {
      setError(err.detail || 'Failed to change password');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearHistory = async () => {
    if (confirmAction !== 'clearHistory') {
      setConfirmAction('clearHistory');
      setTimeout(() => setConfirmAction(null), 5000);
      return;
    }
    
    setIsLoading(true);
    try {
      // await api.clearChatHistory();
      setMessage('Chat history cleared successfully');
      setConfirmAction(null);
    } catch (err) {
      setError('Failed to clear chat history');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadData = async () => {
    setIsLoading(true);
    try {
      // await api.downloadUserData();
      setMessage('Data download will begin shortly');
    } catch (err) {
      setError('Failed to download data');
    } finally {
      setIsLoading(false);
    }
  };

  const handleBackToChat = () => {
    navigate('/chat');
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      <div className="max-w-4xl mx-auto">
        <SettingsHeader handleBackToChat={handleBackToChat} />
        
        <MessageDisplay message={message} error={error} />

        <div className="flex flex-col lg:flex-row gap-6">
          <SettingsTabs activeTab={activeTab} setActiveTab={setActiveTab} />
          
          <SettingsContent
            activeTab={activeTab}
            settings={settings}
            handleSettingChange={handleSettingChange}
            // Account tab props
            handleSaveProfile={handleSaveProfile}
            isProfileLoading={isProfileLoading}
            isLoading={isLoading}
            currentPassword={currentPassword}
            setCurrentPassword={setCurrentPassword}
            newPassword={newPassword}
            setNewPassword={setNewPassword}
            confirmPassword={confirmPassword}
            setConfirmPassword={setConfirmPassword}
            showPasswords={showPasswords}
            setShowPasswords={setShowPasswords}
            handleChangePassword={handleChangePassword}
            email={email}
            setEmail={setEmail}
            handleResetPassword={handleResetPassword}
            // Privacy tab props
            handleDownloadData={handleDownloadData}
            handleClearHistory={handleClearHistory}
            confirmAction={confirmAction}
            // Appearance tab props
            setMessage={setMessage}
            setError={setError}
          />
        </div>
      </div>
    </div>
  );
};