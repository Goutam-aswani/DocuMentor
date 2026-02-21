import { Eye, EyeOff } from 'lucide-react';

export const AccountTab = ({
  settings,
  handleSettingChange,
  handleSaveProfile,
  isProfileLoading,
  isLoading,
  // Password change props
  currentPassword,
  setCurrentPassword,
  newPassword,
  setNewPassword,
  confirmPassword,
  setConfirmPassword,
  showPasswords,
  setShowPasswords,
  handleChangePassword,
  // Password reset props
  email,
  setEmail,
  handleResetPassword
}) => {
  return (
    <div className="space-y-6">
      {/* Profile Information */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white border-b border-gray-600 pb-2">Profile Information</h3>
        {isProfileLoading ? (
          <p className="text-gray-400">Loading profile...</p>
        ) : (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Name</label>
              <input
                type="text"
                value={settings.full_name}
                onChange={(e) => handleSettingChange('full_name', e.target.value)}
                placeholder="Enter your name"
                className="w-full px-4 py-2 text-white bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
              <input
                type="email"
                value={settings.email}
                onChange={(e) => handleSettingChange('email', e.target.value)}
                placeholder="you@example.com"
                className="w-full px-4 py-2 text-white bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="pt-2">
              <button
                onClick={handleSaveProfile}
                disabled={isLoading}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-500"
              >
                {isLoading ? 'Saving...' : 'Save Profile'}
              </button>
            </div>
          </>
        )}
      </div>

      {/* Change Password */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white border-b border-gray-600 pb-2">Change Password</h3>
        <div className="space-y-4">
          <div className="relative">
            <label className="block text-sm font-medium text-gray-300 mb-2">Current Password</label>
            <input
              type={showPasswords.current ? 'text' : 'password'}
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              className="w-full px-4 py-2 pr-10 text-white bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              type="button"
              onClick={() => setShowPasswords(prev => ({ ...prev, current: !prev.current }))}
              className="absolute right-3 top-9 text-gray-400 hover:text-white"
            >
              {showPasswords.current ? <EyeOff size={18} /> : <Eye size={18} />}
            </button>
          </div>
          <div className="relative">
            <label className="block text-sm font-medium text-gray-300 mb-2">New Password</label>
            <input
              type={showPasswords.new ? 'text' : 'password'}
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="w-full px-4 py-2 pr-10 text-white bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              type="button"
              onClick={() => setShowPasswords(prev => ({ ...prev, new: !prev.new }))}
              className="absolute right-3 top-9 text-gray-400 hover:text-white"
            >
              {showPasswords.new ? <EyeOff size={18} /> : <Eye size={18} />}
            </button>
          </div>
          <div className="relative">
            <label className="block text-sm font-medium text-gray-300 mb-2">Confirm New Password</label>
            <input
              type={showPasswords.confirm ? 'text' : 'password'}
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="w-full px-4 py-2 pr-10 text-white bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              type="button"
              onClick={() => setShowPasswords(prev => ({ ...prev, confirm: !prev.confirm }))}
              className="absolute right-3 top-9 text-gray-400 hover:text-white"
            >
              {showPasswords.confirm ? <EyeOff size={18} /> : <Eye size={18} />}
            </button>
          </div>
          <button
            onClick={handleChangePassword}
            disabled={isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-500"
          >
            {isLoading ? 'Updating...' : 'Change Password'}
          </button>
        </div>
      </div>

      {/* Reset Password */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white border-b border-gray-600 pb-2">Reset Password</h3>
        <p className="text-sm text-gray-400">
          Enter your email address below to receive a password reset token.
        </p>
        <div className="space-y-4">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Your email address"
            className="w-full px-4 py-2 text-white bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleResetPassword}
            disabled={isLoading}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:bg-gray-500"
          >
            {isLoading ? 'Sending...' : 'Send Reset Link'}
          </button>
        </div>
      </div>
    </div>
  );
};