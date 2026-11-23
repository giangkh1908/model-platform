import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";

export default [
  { ignores: ["dist/", "node_modules/"] },
  js.configs.recommended,
  {
    files: ["**/*.{js,jsx}"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: globals.browser,
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // Không có plugin react nên tên chỉ dùng trong JSX (vd. `icon: Icon`) bị coi là chưa dùng: bỏ qua tên viết hoa.
      // ignoreRestSiblings: `const { mode, ...rest } = x` là cách cố ý bỏ một khoá.
      "no-unused-vars": ["error", { varsIgnorePattern: "^[A-Z_]", argsIgnorePattern: "^(_|[A-Z])", ignoreRestSiblings: true }],
    },
  },
];
