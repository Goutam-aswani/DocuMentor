import { AccountTab } from './AccountTab';
import { ChatTab } from './ChatTab';
import { NotificationsTab } from './NotificationsTab';
import { PrivacyTab } from './PrivacyTab';
import { AppearanceTab } from './AppearanceTab';

export const SettingsContent = ({ 
  activeTab, 
  settings, 
  handleSettingChange,
  // Account tab props
  handleSaveProfile,
  isProfileLoading,
  isLoading,
  currentPassword,
  setCurrentPassword,
  newPassword,
  setNewPassword,
  confirmPassword,
  setConfirmPassword,
  showPasswords,
  setShowPasswords,
  handleChangePassword,
  email,
  setEmail,
  handleResetPassword,
  // Privacy tab props
  handleDownloadData,
  handleClearHistory,
  confirmAction,
  // Appearance tab props
  setMessage,
  setError
}) => {
  const renderTabContent = () => {
    switch (activeTab) {
      case 'account':
        return (
          <AccountTab
            settings={settings}
            handleSettingChange={handleSettingChange}
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
          />
        );

      case 'chat':
        return (
          <ChatTab
            settings={settings}
            handleSettingChange={handleSettingChange}
          />
        );

      case 'notifications':
        return (
          <NotificationsTab
            settings={settings}
            handleSettingChange={handleSettingChange}
          />
        );

      case 'privacy':
        return (
          <PrivacyTab
            settings={settings}
            handleSettingChange={handleSettingChange}
            handleDownloadData={handleDownloadData}
            handleClearHistory={handleClearHistory}
            isLoading={isLoading}
            confirmAction={confirmAction}
          />
        );

      case 'appearance':
        return (
          <AppearanceTab
            settings={settings}
            handleSettingChange={handleSettingChange}
            isLoading={isLoading}
            setMessage={setMessage}
            setError={setError}
          />
        );

      default:
        return null;
    }
  };

  return (
    <div className="flex-1">
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-6">
        {renderTabContent()}
      </div>
    </div>
  );
};