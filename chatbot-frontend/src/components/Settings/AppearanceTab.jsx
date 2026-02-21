import { Select } from '../ui/Select';

export const AppearanceTab = ({ settings, handleSettingChange, isLoading, setMessage, setError }) => {
  const handleSaveAppearanceSettings = () => {
    // Save theme preference to local storage when button is clicked
    localStorage.setItem('themePreference', settings.theme);
    // You can add logic here to save other appearance settings as well
    setMessage('Appearance settings saved!'); // Provide feedback to the user
    setError(''); // Clear any previous errors
  };

  return (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Theme</label>
        <Select
          value={settings.theme}
          onChange={(value) => handleSettingChange('theme', value)}
          options={[
            { value: 'dark', label: '🌙 Dark' },
            { value: 'light', label: '☀️ Light' },
            { value: 'auto', label: '🖥️ System' }
          ]}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Font Size</label>
        <Select
          value={settings.fontSize}
          onChange={(value) => handleSettingChange('fontSize', value)}
          options={[
            { value: 'small', label: 'Small' },
            { value: 'medium', label: 'Medium' },
            { value: 'large', label: 'Large' },
            { value: 'extra-large', label: 'Extra Large' }
          ]}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">Chat Background</label>
        <Select
          value={settings.chatBackground}
          onChange={(value) => handleSettingChange('chatBackground', value)}
          options={[
            { value: 'default', label: 'Default' },
            { value: 'gradient', label: 'Gradient' },
            { value: 'pattern', label: 'Pattern' },
            { value: 'image', label: 'Custom Image' }
          ]}
        />
      </div>

      <div className="pt-4 border-t border-gray-600">
        <h3 className="text-lg font-semibold text-white mb-4">Preview</h3>
        <div className="p-4 bg-gray-700 rounded-lg border border-gray-600">
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
              <span className="text-sm font-medium">AI</span>
            </div>
            <div className="flex-1">
              <div className={`p-3 rounded-lg bg-gray-600 text-${settings.fontSize === 'small' ? 'sm' : settings.fontSize === 'large' ? 'lg' : 'base'}`}>
                This is how your messages will appear with the current settings.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Save Changes Button */}
      <div className="flex justify-end pt-4">
        <button
          onClick={handleSaveAppearanceSettings}
          disabled={isLoading}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-500"
        >
          Save Changes
        </button>
      </div>
    </div>
  );
};