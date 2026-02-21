import { Download, Trash2 } from 'lucide-react';
import { ToggleSwitch } from '../ui/ToggleSwitch';

export const PrivacyTab = ({ 
  settings, 
  handleSettingChange, 
  handleDownloadData, 
  handleClearHistory, 
  isLoading, 
  confirmAction 
}) => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between py-2">
        <div>
          <span className="text-white font-medium">Two-Factor Authentication</span>
          <p className="text-sm text-gray-400">Add extra security to your account</p>
        </div>
        <ToggleSwitch
          checked={settings.twoFactorAuth}
          onChange={(checked) => handleSettingChange('twoFactorAuth', checked)}
        />
      </div>

      <div className="flex items-center justify-between py-2">
        <div>
          <span className="text-white font-medium">Online Status</span>
          <p className="text-sm text-gray-400">Show when you're online</p>
        </div>
        <ToggleSwitch
          checked={settings.onlineStatus}
          onChange={(checked) => handleSettingChange('onlineStatus', checked)}
        />
      </div>

      <div className="flex items-center justify-between py-2">
        <div>
          <span className="text-white font-medium">Read Receipts</span>
          <p className="text-sm text-gray-400">Let others know you've read messages</p>
        </div>
        <ToggleSwitch
          checked={settings.readReceipts}
          onChange={(checked) => handleSettingChange('readReceipts', checked)}
        />
      </div>

      <div className="flex items-center justify-between py-2">
        <div>
          <span className="text-white font-medium">Last Seen</span>
          <p className="text-sm text-gray-400">Show when you were last active</p>
        </div>
        <ToggleSwitch
          checked={settings.lastSeen}
          onChange={(checked) => handleSettingChange('lastSeen', checked)}
        />
      </div>

      <div className="space-y-4 pt-4 border-t border-gray-600">
        <h3 className="text-lg font-semibold text-white">Data Management</h3>
        
        <button
          onClick={handleDownloadData}
          disabled={isLoading}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-500"
        >
          <Download size={18} />
          <span>Download My Data</span>
        </button>

        <button
          onClick={handleClearHistory}
          disabled={isLoading}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors disabled:bg-gray-500 ${
            confirmAction === 'clearHistory'
              ? 'bg-red-700 text-white'
              : 'bg-red-600 text-white hover:bg-red-700'
          }`}
        >
          <Trash2 size={18} />
          <span>
            {confirmAction === 'clearHistory' ? 'Click again to confirm' : 'Clear Chat History'}
          </span>
        </button>
      </div>
    </div>
  );
};